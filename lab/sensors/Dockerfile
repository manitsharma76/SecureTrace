# Sensor image: tcpdump capture (already used by the lab), plus offline batch extraction run
# AFTER a session: CICFlowMeter (the one true ML feature extractor, see CLAUDE.md "One flow
# extractor") and Zeek (dashboard only, per DEVELOPMENT_PLAN.md -- never a model input).
#
# CICFlowMeter is vendored (lab/sensors/vendor/cicflowmeter, a corrected fork -- see its
# README/LICENSE.txt) and built from source in its own stage; nothing from that stage runs until
# the resulting JRE + jars + native libs are copied into the final Debian-based Zeek image below.
FROM eclipse-temurin:8-jdk-jammy AS cfm-builder
WORKDIR /CICFlowMeter
COPY lab/sensors/vendor/cicflowmeter/ .
# `unzip` flags this build's zip as a (benign) false-positive zip bomb; the original fork's own
# Dockerfile hits the same thing and works around it the same way we do here, just with jar(1)
# (ships with the JDK) instead of pulling in another package.
RUN sed -i 's/\r$//' gradlew \
    && sh ./gradlew --no-daemon distZip \
    && mkdir -p /opt/cicflowmeter-dist \
    && (cd /opt/cicflowmeter-dist && jar xf /CICFlowMeter/build/distributions/CICFlowMeter-*.zip) \
    && mv /opt/cicflowmeter-dist/CICFlowMeter-* /opt/cicflowmeter-dist/app \
    && chmod +x /opt/cicflowmeter-dist/app/bin/cfm /opt/cicflowmeter-dist/app/lib/native/*.so

FROM zeek/zeek:lts AS sensor
RUN echo 'wireshark-common wireshark-common/install-setuid boolean false' | debconf-set-selections \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       tcpdump iproute2 ethtool python3 tshark libpcap0.8 \
    && rm -rf /var/lib/apt/lists/*
# openjdk-8 isn't in Debian 12's repos; use the JRE that came with the builder's JDK image.
COPY --from=cfm-builder /opt/java/openjdk /opt/java/openjdk
COPY --from=cfm-builder /opt/cicflowmeter-dist/app /opt/cicflowmeter
# org.jnetpcap.Pcap's class-level @Library annotation requests the loader load a plain "pcap"
# library (System.loadLibrary("pcap") semantics: looks for libpcap.so on java.library.path, i.e.
# CICFlowMeter's lib/native, not the normal ld.so search path) before touching libjnetpcap*.so
# itself. Put the unversioned symlink there.
RUN ln -s "$(ldconfig -p | awk '/libpcap\.so\.[0-9]/ {print $NF; exit}')" /opt/cicflowmeter/lib/native/libpcap.so
ENV PATH="/opt/java/openjdk/bin:/opt/cicflowmeter/bin:${PATH}"
# libjnetpcap-pcap100.so and libjnetpcap.so depend on each other; java.library.path (set via
# CICFlowMeter's own start script) only resolves the first dlopen(), so their mutual dependency
# still needs the normal dynamic-linker search path.
ENV LD_LIBRARY_PATH="/opt/cicflowmeter/lib/native"
WORKDIR /app
COPY lab/sensors/extract.py lab/sensors/extract.py
COPY lab/sensors/capture.py lab/sensors/capture.py
COPY lab/sensors/zeek_conn.zeek lab/sensors/zeek_conn.zeek
ENTRYPOINT ["sleep", "infinity"]
