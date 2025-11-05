import argparse
import datetime
import json
import random
import time
from typing import Optional, Any
import requests


def make_payload(device_id: str,
                 heartRate: int | None = None,
                 o2Sat: int | None = None,
                 accX: float | None = None,
                 accY: float | None = None,
                 accZ: float | None = None,
                 gyroX: float | None = None,
                 gyroY: float | None = None,
                 gyroZ: float | None = None,
                 skinTemp: float | None = None) -> dict:
    """Generate a payload; accepts explicit values for all data fields.

    If a value is None the function will generate a randomized value using the
    same ranges/rounding as the previous implementation. The timestamp is still
    generated inside the function (UTC, no microseconds, 'Z' suffix).
    """
    # timestamp (kept internal)
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

    # fallback generation when arguments are not provided
    hr = heartRate if heartRate is not None else random.randint(70, 90)
    o2 = o2Sat if o2Sat is not None else random.randint(96, 100)
    ax = accX if accX is not None else round(random.randint(100, 200) / 100.0, 2)
    ay = accY if accY is not None else round(random.randint(200, 300) / 100.0, 2)
    az = accZ if accZ is not None else round(random.randint(300, 400) / 100.0, 2)
    gx = gyroX if gyroX is not None else round(random.randint(100, 200) / 100.0, 2)
    gy = gyroY if gyroY is not None else round(random.randint(200, 300) / 100.0, 2)
    gz = gyroZ if gyroZ is not None else round(random.randint(300, 400) / 100.0, 2)
    st = skinTemp if skinTemp is not None else round(random.uniform(36.5, 37.5), 1)

    payload = {
        "deviceId": device_id,
        "timestamp": timestamp,
        "heartRate": hr,
        "o2Sat": o2,
        "accX": ax,
        "accY": ay,
        "accZ": az,
        "gyroX": gx,
        "gyroY": gy,
        "gyroZ": gz,
        "skinTemp": st,
    }
    return payload


def send_payload(url: str, payload: dict, timeout: float = 5.0) -> Optional[Any]:
    """Send JSON payload to the given URL. Uses requests if available, otherwise falls back to urllib."""
    if requests is None:
        # Minimal fallback using urllib to avoid an external dependency, but requests is preferred.
        try:
            import urllib.request
            import urllib.error

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # emulate a tiny Response-like object
                class _Resp:
                    def __init__(self, code, text):
                        self.status_code = code
                        self.text = text

                return _Resp(resp.getcode(), resp.read().decode('utf-8'))
        except Exception as e:
            print(f"Failed to send (urllib fallback): {e}")
            return None
    else:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            return resp
        except Exception as e:
            print(f"Failed to send (requests): {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Send simulated device data to an HTTP endpoint in a loop.")
    parser.add_argument("--device", "-d", default="123456", help="deviceId to include in payload")
    parser.add_argument("--url", "-u", default="http://192.168.137.1:3000/api/data", help="Endpoint URL")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Seconds between sends")
    args = parser.parse_args()

    print(f"Starting data sender -> {args.url} (device={args.device}, interval={args.interval}s)")
    if requests is None:
        print("Note: 'requests' not available; using urllib fallback. For best results: pip install requests")

    try:
        while True:
            payload = make_payload("aaaaaa", 85 + random.randint(0, 20), 95, 0, 0.1, 0, 0, 0.1, 0, 36.5 + random.random())
            resp = send_payload(args.url, payload)
            if resp is None:
                print("No response (send failed)")
            else:
                # requests.Response has status_code and text; our fallback object mimics these
                print(f"Sent: {json.dumps(payload)} -> status={getattr(resp, 'status_code', None)}")
            time.sleep(args.interval)

            if False:
                payload = make_payload("aaaaaa", 50 + random.randint(0, 20), 91, 5, 3.1, -5, 1, 4, 9, 36.5 + random.random())
                resp = send_payload(args.url, payload)
                if resp is None:
                    print("No response (send failed)")
                else:
                    # requests.Response has status_code and text; our fallback object mimics these
                    print(f"Sent: {json.dumps(payload)} -> status={getattr(resp, 'status_code', None)}")
                time.sleep(args.interval)


    except KeyboardInterrupt:
        print('\nStopped by user')


if __name__ == '__main__':
    main()
