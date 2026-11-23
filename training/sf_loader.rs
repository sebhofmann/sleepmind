//! Loader for Stockfish binpacks (external data, e.g. the Leela-derived test80
//! sets relabelled with BT4, see download_leela_data.sh).
//!
//! Differences to bullet's `SfBinpackLoader`:
//!   * all files are read at the same time: every chunk comes from a file drawn
//!     with probability proportional to its remaining bytes, so the months/sources
//!     are mixed evenly instead of being trained one after another;
//!   * persistent worker threads, so the thread-local filter state (piece-count
//!     statistics, placeholder-zero detection) works like in the reference;
//!   * the position filter and the score target follow the reference trainer
//!     used by vondele/nettest (xu-shawn/nnue-pytorch 289991e, threats.yaml):
//!     data_loader/cpp/training_data_loader.cpp `make_skip_predicate` and
//!     model/nnue.py `calculate_sf_loss`.
//!
//! Score target: the reference turns a data score s (Stockfish units) into
//!   p = 0.5 * (1 + sigmoid((s - OUT_OFFSET) / OUT_SCALING) - sigmoid((-s - OUT_OFFSET) / OUT_SCALING)).
//! Our trainer uses sigmoid(score / EVAL_SCALE), so each score is replaced by
//! EVAL_SCALE * logit(p): the target probability is exactly the reference's and the
//! network output keeps the units of our engine (NNUE_SCALE).

use std::{
    fs::File,
    io::BufReader,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
};

use bullet::game::formats::{
    bulletformat::ChessBoard,
    sfbinpack::{
        chess::{color::Color, piece::Piece, piecetype::PieceType},
        read_chunk_into, ChunkReader, TrainingDataEntry,
    },
};
use bullet_trainer::reader::DataReader;
use rand::Rng;

// ---- reference settings (nettest threats.yaml) ------------------------------
const VALUE_NONE: i16 = 32002;
const RANDOM_FEN_SKIPPING: f64 = 3.0; // keeps 1 / (3 + 1)
const EARLY_FEN_SKIPPING: u16 = 18; // ply <= 18 dropped
const SOFT_EARLY_FEN_SKIPPING: u16 = 32; // ply < 32 accepted with the curve below
const PLY_CURVE: [(f64, f64); 4] = [(0.0, 0.025), (22.0, 0.05), (25.5, 0.20), (29.5, 0.80)];
const PC_Y: [f64; 5] = [-0.20, 0.45, 1.0, 0.95, 0.75]; // desired piece-count weights at 0/8/16/24/32
const MAX_PC_SKIP_RATE: f64 = 0.975;
const OUT_SCALING: f64 = 379.98724077106635;
const OUT_OFFSET: f64 = 289.1258344152218;
// Piece-count resampling discards many positions; switch off to keep more data.
pub const PIECE_COUNT_FILTER: bool = true;

// Pipeline sizes as in the reference (data_loader/cpp/lib/parallel_dataloader.h):
// workers decode one chunk per claim, fill shuffle windows of 2^20 positions, and
// up to 16 sealed windows wait for the trainer. Small windows keep the flow steady
// instead of long decode bursts followed by idle phases.
const CHUNKS_PER_JOB: usize = 1;
const CHUNK_QUEUE: usize = 256;
pub const SHUFFLE_WINDOW: usize = 1 << 20;
const READY_WINDOWS: usize = 16;

#[derive(Clone)]
pub struct SfLoader {
    paths: Vec<String>,
    threads: usize,
    eval_scale: f32,
}

impl SfLoader {
    pub fn new(paths: &[String], threads: usize, eval_scale: f32) -> Self {
        for path in paths {
            assert!(std::path::Path::new(path).exists(), "File not found: {path}");
        }
        Self {
            paths: paths.to_vec(),
            threads: threads.max(1),
            eval_scale,
        }
    }

    /// Reads `max_chunks` chunks spread over all files and prints the filter
    /// pass rate and the expected number of training positions per epoch.
    pub fn print_stats(&self, max_chunks: usize) {
        let total_bytes: u64 = self.paths.iter().map(|p| std::fs::metadata(p).unwrap().len()).sum();
        let per_file = max_chunks.div_ceil(self.paths.len()).max(1);
        let (mut bytes, mut raw, mut kept) = (0u64, 0u64, 0u64);
        let mut filter = Filter::new();
        let mut buf = Vec::new();
        for path in &self.paths {
            let mut reader = BufReader::with_capacity(1 << 20, File::open(path).unwrap());
            for _ in 0..per_file {
                if !read_chunk_into(&mut reader, &mut buf).unwrap() {
                    break;
                }
                bytes += buf.len() as u64 + 8;
                let mut cr = ChunkReader::default();
                while cr.has_next(&buf) {
                    let e = cr.next(&buf);
                    raw += 1;
                    kept += u64::from(!filter.skip(&e));
                }
            }
        }
        let raw_total = raw as f64 / bytes as f64 * total_bytes as f64;
        let pass = kept as f64 / raw.max(1) as f64;
        println!("sfbinpack:     {} file(s), {:.1} GB", self.paths.len(), total_bytes as f64 / 1e9);
        println!(
            "               sample {:.0} MB: {raw} positions ({:.2} bytes/pos), filter keeps {:.2} %",
            bytes as f64 / 1e6,
            bytes as f64 / raw.max(1) as f64,
            100.0 * pass
        );
        println!(
            "               -> ~{:.2e} positions in the files, ~{:.2e} training positions per pass",
            raw_total,
            raw_total * pass
        );
    }
}

impl DataReader<ChessBoard> for SfLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _skip: usize, mut f: F) {
        let stop = Arc::new(AtomicBool::new(false));

        // 1. file reader: draws the next chunk from a random file (weighted by remaining bytes)
        let (job_tx, job_rx) = mpsc::sync_channel::<Vec<Vec<u8>>>(CHUNK_QUEUE / CHUNKS_PER_JOB);
        let paths = self.paths.clone();
        let reader_stop = stop.clone();
        std::thread::spawn(move || {
            let mut rng = rand::rng();
            let mut epoch = 0u64;
            'outer: loop {
                epoch += 1;
                let mut files: Vec<(BufReader<File>, u64)> = paths
                    .iter()
                    .map(|p| {
                        let len = std::fs::metadata(p).unwrap().len();
                        (BufReader::with_capacity(1 << 22, File::open(p).unwrap()), len)
                    })
                    .collect();
                let mut job = Vec::with_capacity(CHUNKS_PER_JOB);
                while !files.is_empty() {
                    let total: u64 = files.iter().map(|(_, rem)| *rem).sum();
                    let mut pick = rng.random_range(0..total.max(1));
                    let mut idx = 0;
                    while idx + 1 < files.len() && pick >= files[idx].1 {
                        pick -= files[idx].1;
                        idx += 1;
                    }
                    let mut chunk = Vec::new();
                    if read_chunk_into(&mut files[idx].0, &mut chunk).unwrap_or(false) {
                        files[idx].1 = files[idx].1.saturating_sub(chunk.len() as u64 + 8);
                        job.push(chunk);
                        if job.len() == CHUNKS_PER_JOB {
                            if reader_stop.load(Ordering::Relaxed) || job_tx.send(std::mem::take(&mut job)).is_err() {
                                break 'outer;
                            }
                        }
                    } else {
                        files.swap_remove(idx);
                    }
                }
                if !job.is_empty() && job_tx.send(job).is_err() {
                    break;
                }
                if epoch == 1 {
                    println!("sfbinpack: first pass over all files finished");
                }
            }
        });

        // 2. workers: unpack, filter, convert
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (pos_tx, pos_rx) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let eval_scale = self.eval_scale;
        for _ in 0..self.threads {
            let job_rx = job_rx.clone();
            let pos_tx = pos_tx.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut filter = Filter::new();
                loop {
                    let job = match job_rx.lock().unwrap().recv() {
                        Ok(job) => job,
                        Err(_) => break,
                    };
                    let mut out = Vec::with_capacity(job.len() * 32768);
                    for chunk in &job {
                        let mut cr = ChunkReader::default();
                        while cr.has_next(chunk) {
                            let e = cr.next(chunk);
                            if !filter.skip(&e) {
                                out.push(convert(&e, eval_scale));
                            }
                        }
                    }
                    if stop.load(Ordering::Relaxed) || pos_tx.send(out).is_err() {
                        break;
                    }
                }
            });
        }
        drop(pos_tx);

        // 3. shuffle windows: filled and shuffled on their own thread, handed over FIFO
        let (buf_tx, buf_rx) = mpsc::sync_channel::<Vec<ChessBoard>>(READY_WINDOWS);
        std::thread::spawn(move || {
            let mut window = Vec::with_capacity(SHUFFLE_WINDOW);
            while let Ok(positions) = pos_rx.recv() {
                for p in positions {
                    window.push(p);
                    if window.len() == SHUFFLE_WINDOW {
                        shuffle(&mut window);
                        if buf_tx.send(std::mem::replace(&mut window, Vec::with_capacity(SHUFFLE_WINDOW))).is_err() {
                            return;
                        }
                    }
                }
            }
        });

        while let Ok(buffer) = buf_rx.recv() {
            if f(&buffer) {
                break;
            }
        }
        stop.store(true, Ordering::Relaxed);
        // receivers are dropped here, the producers' next send fails and they exit
    }
}

/// The reference's skip predicate with its thread-local state.
struct Filter {
    last_ply: i32,
    last_score: i16,
    ply_accept: [f64; SOFT_EARLY_FEN_SKIPPING as usize + 1],
    pc_target: [f64; 33],
    pc_target_total: f64,
    pc_all: [f64; 33],
    pc_all_total: f64,
    alpha: f64,
    steps: u64,
}

impl Filter {
    fn new() -> Self {
        let mut ply_accept = [1.0; SOFT_EARLY_FEN_SKIPPING as usize + 1];
        let pts: Vec<(f64, f64)> =
            PLY_CURVE.iter().copied().chain(std::iter::once((f64::from(SOFT_EARLY_FEN_SKIPPING), 1.0))).collect();
        for (ply, v) in ply_accept.iter_mut().enumerate() {
            let x = ply as f64;
            *v = if x <= pts[0].0 {
                pts[0].1
            } else if x >= pts[4].0 {
                pts[4].1
            } else {
                let i = (0..4).find(|&i| x >= pts[i].0 && x <= pts[i + 1].0).unwrap();
                let t = (x - pts[i].0) / (pts[i + 1].0 - pts[i].0);
                pts[i].1 + t * (pts[i + 1].1 - pts[i].1)
            }
            .clamp(0.0, 1.0);
        }

        let mut pc_target = [0.0; 33];
        for (pc, w) in pc_target.iter_mut().enumerate() {
            *w = pc_weight(pc as f64);
        }
        let pc_target_total = pc_target.iter().sum();

        Self {
            last_ply: -1,
            last_score: VALUE_NONE,
            ply_accept,
            pc_target,
            pc_target_total,
            pc_all: [0.0; 33],
            pc_all_total: 0.0,
            alpha: 1.0,
            steps: 0,
        }
    }

    /// true = drop the position
    fn skip(&mut self, e: &TrainingDataEntry) -> bool {
        let mut rng = rand::rng();

        // placeholder score 0 inside a decisive game after a large score
        let placeholder_zero = i32::from(e.ply) > self.last_ply
            && self.last_score != VALUE_NONE
            && self.last_score.unsigned_abs() > 100
            && e.result != 0
            && e.score == 0;
        self.last_ply = i32::from(e.ply);
        if e.score == VALUE_NONE || placeholder_zero {
            return true;
        }
        self.last_score = e.score;

        if e.ply <= EARLY_FEN_SKIPPING {
            return true;
        }
        if rng.random::<f64>() < RANDOM_FEN_SKIPPING / (RANDOM_FEN_SKIPPING + 1.0) {
            return true;
        }
        if is_capturing_move(e) || e.pos.is_checked(e.pos.side_to_move()) {
            return true;
        }
        if e.ply < SOFT_EARLY_FEN_SKIPPING && rng.random::<f64>() >= self.ply_accept[e.ply as usize] {
            return true;
        }
        if !PIECE_COUNT_FILTER {
            return false;
        }

        // dynamic piece-count resampling towards pc_target
        let pc = e.pos.occupied().count() as usize;
        if pc > 32 {
            return true;
        }
        self.pc_all[pc] += 1.0;
        self.pc_all_total += 1.0;
        self.steps += 1;
        let s = self.steps;
        if matches!(s, 100 | 500 | 1000 | 2500 | 5000) || (s > 5000 && s % 10000 == 0) {
            let min_ratio = (0..33)
                .filter(|&i| self.pc_target[i] > 0.0 && self.pc_all[i] > 0.0)
                .map(|i| self.ratio(i))
                .fold(f64::INFINITY, f64::min);
            if min_ratio.is_finite() && min_ratio > 0.0 {
                self.alpha = (1.0 - MAX_PC_SKIP_RATE) / min_ratio;
            }
            if s >= 10000 && s % 10000 == 0 {
                self.pc_all.iter_mut().for_each(|c| *c *= 0.5);
                self.pc_all_total *= 0.5;
            }
        }
        let accept = if self.pc_target[pc] > 0.0 && self.pc_all[pc] > 0.0 {
            (self.alpha * self.ratio(pc)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        rng.random::<f64>() >= accept
    }

    fn ratio(&self, pc: usize) -> f64 {
        (self.pc_all_total * self.pc_target[pc]) / (self.pc_target_total * self.pc_all[pc])
    }
}

/// Cubic Hermite spline through PC_Y at piece counts 0, 8, 16, 24, 32 (reference LUT).
fn pc_weight(x: f64) -> f64 {
    let y = PC_Y;
    if x <= 0.0 {
        return y[0];
    }
    if x >= 32.0 {
        return y[4];
    }
    let i = ((x / 8.0) as usize).min(3);
    let t = (x - i as f64 * 8.0) / 8.0;
    let slope = |idx: usize| match idx {
        0 => y[1] - y[0],
        4 => y[4] - y[3],
        _ => (y[idx + 1] - y[idx - 1]) / 2.0,
    };
    let (m0, m1) = (slope(i), slope(i + 1));
    let (t2, t3) = (t * t, t * t * t);
    let val = (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
        + (t3 - t2) * m1;
    val.max(0.0)
}

/// Same as the reference's isCapturingMove: castling (king takes own rook) is
/// not a capture, en passant is not counted either.
fn is_capturing_move(e: &TrainingDataEntry) -> bool {
    let to = e.pos.piece_at(e.mv.to());
    to != Piece::none() && to.color() != e.pos.piece_at(e.mv.from()).color()
}

/// Reference target probability of a Stockfish score, as our score in EVAL_SCALE units.
fn sf_score_to_ours(score: i16, eval_scale: f32) -> i16 {
    let s = f64::from(score);
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let p = 0.5 * (1.0 + sig((s - OUT_OFFSET) / OUT_SCALING) - sig((-s - OUT_OFFSET) / OUT_SCALING));
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (f64::from(eval_scale) * (p / (1.0 - p)).ln()).round() as i16
}

/// Like bullet's sfbinpack conversion, with the score mapped by `sf_score_to_ours`.
fn convert(e: &TrainingDataEntry, eval_scale: f32) -> ChessBoard {
    let stm = usize::from(e.pos.side_to_move().ordinal());
    let pc_bb = |pt| e.pos.pieces_bb_color(Color::Black, pt).bits() | e.pos.pieces_bb_color(Color::White, pt).bits();
    let bbs = [
        e.pos.pieces_bb(Color::White).bits(),
        e.pos.pieces_bb(Color::Black).bits(),
        pc_bb(PieceType::Pawn),
        pc_bb(PieceType::Knight),
        pc_bb(PieceType::Bishop),
        pc_bb(PieceType::Rook),
        pc_bb(PieceType::Queen),
        pc_bb(PieceType::King),
    ];
    let mut score = sf_score_to_ours(e.score, eval_scale);
    let mut result = f32::from(1 + e.result) / 2.0;
    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }
    ChessBoard::from_raw(bbs, stm, score, result).expect("malformed binpack entry")
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = rand::rng();
    for i in (1..data.len()).rev() {
        data.swap(i, rng.random_range(0..=i));
    }
}
