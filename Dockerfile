FROM lichessbotdevs/lichess-bot:2026.11.16.2
COPY . /opt/sleepmind
WORKDIR /opt/sleepmind
RUN ls -la
RUN make both && cp ./build/sleepmind /usr/local/bin/sleepmind
WORKDIR /lichess-bot
COPY quantised_threats.bin /usr/local/bin/quantised_threats.bin
RUN chmod +x /usr/local/bin/sleepmind

CMD ["/bin/sh", "-c", "python3 lichess-bot.py --config /config.yaml"]