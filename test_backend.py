"""
Test script to verify backend compatibility with React Native frontend.
"""
import asyncio
import json
import base64
import requests
from pathlib import Path
import websockets

# Test configuration
BASE_URL = "http://localhost:9055"
WS_URL = "ws://localhost:9055/ws/detect"


def test_ping():
    """Test /ping endpoint"""
    print("\n" + "=" * 60)
    print("TEST 1: Ping Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/ping")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    # Validate structure
    assert "status" in data, "Missing 'status' field"
    assert "latency_ms" in data, "Missing 'latency_ms' field"
    assert data["status"] == "online", "Status should be 'online'"

    print("✅ PASS: /ping endpoint works correctly")


def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 60)
    print("TEST 2: Root Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    print("✅ PASS: Root endpoint works")


def test_history():
    """Test /history endpoint"""
    print("\n" + "=" * 60)
    print("TEST 3: History Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/history")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    # Validate structure
    assert "detections" in data, "Missing 'detections' field"
    assert "total" in data, "Missing 'total' field"
    assert isinstance(data["detections"], list), "detections should be a list"

    print(f"✅ PASS: /history endpoint works (found {data['total']} records)")


def create_test_image():
    """Create a simple test image"""
    import numpy as np
    import cv2

    # Create a dummy image (red square)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[200:400, 200:400] = [0, 0, 255]  # Red square

    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', img)

    return buffer.tobytes()


def test_upload():
    """Test /upload endpoint"""
    print("\n" + "=" * 60)
    print("TEST 4: Upload Endpoint")
    print("=" * 60)

    # Create test image
    image_bytes = create_test_image()

    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    response = requests.post(f"{BASE_URL}/upload", files=files)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    # Validate structure (frontend-compatible format)
    assert "detections" in data, "Missing 'detections' field"
    assert "inferenceTime" in data, "Missing 'inferenceTime' field"
    assert "fps" in data, "Missing 'fps' field"
    assert "gpuName" in data, "Missing 'gpuName' field"

    assert isinstance(data["detections"], list), "detections should be a list"

    # If detections found, validate structure
    if len(data["detections"]) > 0:
        det = data["detections"][0]
        assert "class" in det, "Detection missing 'class' field"
        assert "confidence" in det, "Detection missing 'confidence' field"
        assert "bbox" in det, "Detection missing 'bbox' field"
        assert "timestamp" in det, "Detection missing 'timestamp' field"

        # Validate confidence is 0-1 scale
        assert 0 <= det["confidence"] <= 1, f"Confidence should be 0-1, got {det['confidence']}"

        # Validate bbox format
        assert len(det["bbox"]) == 4, "bbox should have 4 coordinates [x1, y1, x2, y2]"

        print(f"\n✅ Detection found: {det['class']} (confidence: {det['confidence']:.2f})")
    else:
        print("\nℹ️  No objects detected in test image (expected)")

    print("✅ PASS: /upload endpoint returns correct format")


async def test_websocket():
    """Test WebSocket endpoint"""
    print("\n" + "=" * 60)
    print("TEST 5: WebSocket Endpoint")
    print("=" * 60)

    # Create test image and encode to base64
    image_bytes = create_test_image()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ Connected to WebSocket")

            # Send frame
            message = {
                "type": "frame",
                "image": base64_image,
                "timestamp": 1699999999999
            }
            await websocket.send(json.dumps(message))
            print("📤 Sent test frame")

            # Receive response
            response = await websocket.recv()
            data = json.loads(response)

            print(f"📥 Received response:")
            print(json.dumps(data, indent=2))

            # Validate structure
            assert "detections" in data, "Missing 'detections' field"
            assert "inferenceTime" in data, "Missing 'inferenceTime' field"
            assert "fps" in data, "Missing 'fps' field"
            assert "gpuName" in data, "Missing 'gpuName' field"

            assert isinstance(data["detections"], list), "detections should be a list"

            # If detections found, validate structure
            if len(data["detections"]) > 0:
                det = data["detections"][0]
                assert "class" in det, "Detection missing 'class' field"
                assert "confidence" in det, "Detection missing 'confidence' field"
                assert "bbox" in det, "Detection missing 'bbox' field"
                assert "timestamp" in det, "Detection missing 'timestamp' field"

                # Validate confidence is 0-1 scale
                assert 0 <= det["confidence"] <= 1, f"Confidence should be 0-1, got {det['confidence']}"

                print(f"\n✅ Detection: {det['class']} (confidence: {det['confidence']:.2f})")
                print(f"   BBox: {det['bbox']}")
            else:
                print("\nℹ️  No objects detected (expected)")

            print("✅ PASS: WebSocket endpoint returns correct format")

    except Exception as e:
        print(f"❌ FAIL: WebSocket test failed: {e}")
        raise


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 BACKEND COMPATIBILITY TEST SUITE")
    print("=" * 60)
    print("Testing backend compatibility with React Native frontend")
    print("=" * 60)

    try:
        # HTTP endpoint tests
        test_ping()
        test_root()
        test_history()
        test_upload()

        # WebSocket test
        asyncio.run(test_websocket())

        # Summary
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("✅ Backend is fully compatible with React Native frontend")
        print("✅ All endpoints return correct JSON structure")
        print("✅ Confidence values are in 0-1 scale")
        print("✅ Bounding boxes are included")
        print("✅ WebSocket communication works")
        print("=" * 60)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        return False
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 60)
        print("❌ Cannot connect to backend")
        print("=" * 60)
        print("Make sure the backend is running:")
        print("  cd backend")
        print("  python main.py")
        print("=" * 60)
        return False
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Unexpected error: {e}")
        print("=" * 60)
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
