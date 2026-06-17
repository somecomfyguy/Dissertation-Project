#!/bin/bash
# =============================================================================
# test_pluto_hw.sh  —  PlutoSDR (clone) Hardware Verification on Jetson Nano
# =============================================================================
# Checks the full driver stack: USB visibility → libiio IIO context →
# network reachability → Python libiio bindings → gr-iio import.
# Run as the normal user (not root) after plugging in the PlutoSDR via USB.
#
# Usage:
#   chmod +x test_pluto_hw.sh
#   ./test_pluto_hw.sh
#
# Expected outcome for a healthy setup:
#   All PASS lines, URI printed, IP reachable, Python imports succeed.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "${GRN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
info() { echo -e "${YLW}[INFO]${NC} $1"; }

echo "============================================================"
echo "  PlutoSDR Hardware Verification — Jetson Nano"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# 1. USB device visibility
# ------------------------------------------------------------
echo "--- Step 1: USB visibility ---"
if lsusb | grep -iq "Analog Devices\|0456:b673\|04b4:00f1\|2500:0020"; then
    pass "PlutoSDR USB device found:"
    lsusb | grep -i "Analog Devices\|0456:b673\|04b4:00f1\|2500:0020" | sed 's/^/         /'
else
    info "Analog Devices VID not matched — listing all USB devices for manual check:"
    lsusb | sed 's/^/         /'
    echo ""
    info "PlutoSDR clones sometimes use different VID/PIDs. Check the list above."
    info "The device should appear when plugged in and disappear when removed."
fi
echo ""

# ------------------------------------------------------------
# 2. libiio installation
# ------------------------------------------------------------
echo "--- Step 2: libiio installation ---"
if command -v iio_info &> /dev/null; then
    IIO_VER=$(iio_info --version 2>/dev/null | head -1 || echo "unknown")
    pass "iio_info found: $IIO_VER"
else
    fail "iio_info not found — install libiio:"
    echo "       sudo apt-get install libiio-utils libiio-dev"
    echo "       # or build from source: https://github.com/analogdevicesinc/libiio"
fi
echo ""

# ------------------------------------------------------------
# 3. IIO context scan (USB backend)
# ------------------------------------------------------------
echo "--- Step 3: IIO context scan (USB) ---"
info "Scanning for IIO devices (USB backend)..."
IIO_SCAN=$(iio_info -s 2>/dev/null || true)
if echo "$IIO_SCAN" | grep -q "PlutoSDR\|ad9361\|cf-ad9361\|iio:device"; then
    pass "PlutoSDR IIO context detected via USB scan:"
    echo "$IIO_SCAN" | sed 's/^/         /'
    # Extract URI for later use
    PLUTO_URI=$(echo "$IIO_SCAN" | grep -oP 'usb:[0-9.]+' | head -1 || true)
    if [ -n "$PLUTO_URI" ]; then
        info "Detected URI: $PLUTO_URI"
        export PLUTO_URI
    fi
else
    fail "No PlutoSDR found via USB IIO scan."
    echo "       Output was: $IIO_SCAN"
    echo ""
    info "Troubleshooting:"
    echo "       1. Replug the USB cable."
    echo "       2. Check 'dmesg | tail -20' for USB errors."
    echo "       3. Add udev rule if permission denied:"
    echo '          echo '"'"'SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", MODE="0666", GROUP="plugdev"'"'"' | sudo tee /etc/udev/rules.d/53-adi-plutosdr-usb.rules'
    echo "          sudo udevadm control --reload-rules && sudo udevadm trigger"
fi
echo ""

# ------------------------------------------------------------
# 4. Network reachability (USB RNDIS/ECM — 192.168.2.1)
# ------------------------------------------------------------
echo "--- Step 4: USB network (RNDIS/ECM) reachability ---"
PLUTO_IP="192.168.2.1"
info "Pinging PlutoSDR at $PLUTO_IP (USB Ethernet emulation)..."
if ping -c 2 -W 2 "$PLUTO_IP" &> /dev/null; then
    pass "PlutoSDR reachable at $PLUTO_IP"
    # Prefer IP URI if network is up (more stable than USB backend for GNSS-SDR)
    export PLUTO_URI="ip:$PLUTO_IP"
    info "Will use URI: ip:$PLUTO_IP"
else
    info "Not reachable at $PLUTO_IP — this is OK if using USB URI directly."
    info "To enable IP access, the Jetson needs a USB RNDIS network interface."
    info "Check 'ip addr' for a usb0/eth1 interface with 192.168.2.10."
fi
echo ""

# ------------------------------------------------------------
# 5. IIO device attributes (verify it is an AD9361/AD9363)
# ------------------------------------------------------------
echo "--- Step 5: IIO device attributes ---"
FINAL_URI="${PLUTO_URI:-ip:192.168.2.1}"
info "Querying device attributes at URI: $FINAL_URI"
if iio_info -u "$FINAL_URI" 2>/dev/null | grep -q "ad9361\|ad9363\|cf-ad9361"; then
    pass "AD9361/AD9363 RF chip confirmed via IIO attributes."
    iio_info -u "$FINAL_URI" 2>/dev/null | grep -E "Library|IIO context|description|ad936" | sed 's/^/         /'
else
    info "Could not verify chip via IIO attributes at $FINAL_URI."
    info "Try: iio_info -u $FINAL_URI   (manually inspect output)"
fi
echo ""

# ------------------------------------------------------------
# 6. Python libiio bindings
# ------------------------------------------------------------
echo "--- Step 6: Python libiio bindings ---"
if python3 -c "import iio; ctx = iio.Context('$FINAL_URI'); print('Devices:', [d.name for d in ctx.devices])" 2>/dev/null; then
    pass "Python libiio import and context creation succeeded."
else
    PYIIO_ERR=$(python3 -c "import iio" 2>&1 || true)
    if echo "$PYIIO_ERR" | grep -q "ModuleNotFoundError"; then
        fail "Python libiio (pylibiio) not installed."
        echo "       Install: pip3 install pylibiio"
        echo "       Or build from source (libiio ships Python bindings)."
    else
        fail "Python libiio import failed: $PYIIO_ERR"
    fi
fi
echo ""

# ------------------------------------------------------------
# 7. GNU Radio gr-iio import
# ------------------------------------------------------------
echo "--- Step 7: GNU Radio gr-iio block availability ---"
if python3 -c "from gnuradio import iio; print('gr-iio version:', iio.api_version())" 2>/dev/null; then
    pass "gr-iio available in GNU Radio."
else
    GR_ERR=$(python3 -c "from gnuradio import iio" 2>&1 || true)
    fail "gr-iio import failed: $GR_ERR"
    echo "       Install: sudo apt-get install gr-iio"
    echo "       Verify GNU Radio is installed: gnuradio-config-info --version"
fi
echo ""

# ------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo ""
echo "  Recommended URI for subsequent scripts:"
echo "    PLUTO_URI=${PLUTO_URI:-ip:192.168.2.1}"
echo ""
echo "  Next step:"
echo "    python3 test_pluto_gnuradio.py --uri ${PLUTO_URI:-ip:192.168.2.1}"
echo ""
