#!/usr/bin/env python
"""Test script to verify VibeVox installation and TTS functionality."""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("qwen_tts", "Qwen TTS"),
        ("soundfile", "SoundFile"),
        ("fastapi", "FastAPI"),
        ("httpx", "HTTPX"),
        ("nltk", "NLTK"),
    ]
    
    failed = []
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed imports: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!\n")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("🔧 Testing CUDA...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✓ CUDA available")
            print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA Version: {torch.version.cuda}")
            print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  ⚠️  CUDA not available - will use CPU (slower)")
            print("      Ensure you have CUDA-compatible GPU and drivers installed")
        
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}\n")
        return False


def test_flash_attention():
    """Test Flash Attention availability."""
    print("⚡ Testing Flash Attention...")
    
    try:
        import flash_attn
        print(f"  ✓ Flash Attention installed (version {flash_attn.__version__})")
        print()
        return True
    except ImportError:
        print("  ⚠️  Flash Attention not installed")
        print("      Install with: pip install flash-attn --no-build-isolation")
        print("      TTS server will fall back to eager attention (slower)")
        print()
        return True  # Not critical, just slower


def test_groq_key():
    """Test GROQ API key."""
    print("🔑 Testing GROQ API key...")
    
    import os
    
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print(f"  ✓ GROQ_API_KEY is set ({api_key[:10]}...)")
        print()
        return True
    else:
        print("  ✗ GROQ_API_KEY not set")
        print("      Set with: export GROQ_API_KEY='your-key-here'")
        print()
        return False


def test_model_loading():
    """Test if Qwen3-TTS model can be loaded."""
    print("🎙️ Testing Qwen3-TTS model loading...")
    print("   (This may take a while on first run - downloading ~3.5GB)")
    
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
        
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(f"   Loading {model_id}...")
        print(f"   Device: {device}, Dtype: {dtype}")
        
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation="eager",  # Use eager for testing to avoid flash_attn issues
        )
        
        print("  ✓ Model loaded successfully!")
        
        # Quick test synthesis
        print("\n  Testing voice synthesis...")
        wavs, sr = model.generate_voice_design(
            text="Hello, this is a test of VibeVox.",
            language="English",
            instruct="A clear, neutral male voice with steady pacing.",
        )
        
        print(f"  ✓ Synthesis successful! Generated {len(wavs[0])} samples at {sr}Hz")
        
        # Save test audio
        import soundfile as sf
        test_file = Path("test_tts_output.wav")
        sf.write(test_file, wavs[0], sr)
        print(f"  ✓ Saved test audio to: {test_file.absolute()}")
        print()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection (model needs to download)")
        print("  2. Ensure you have enough disk space (~5GB)")
        print("  3. Check CUDA/GPU availability")
        print()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("VibeVox Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Critical tests
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Flash Attention", test_flash_attention()))
    
    # Check if we should continue with heavy tests
    if not all(r[1] for r in results if r[0] == "Imports"):
        print("❌ Basic imports failed. Fix errors and try again.")
        return 1
    
    results.append(("GROQ API Key", test_groq_key()))
    
    # Ask before model test (downloads large files)
    print("⚠️  The next test will download the Qwen3-TTS model (~3.5GB)")
    response = input("   Continue? [y/N]: ").strip().lower()
    if response == 'y':
        results.append(("Model Loading", test_model_loading()))
    else:
        print("   Skipped model loading test.")
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("🎉 All tests passed! VibeVox is ready to use.")
        print("\nNext steps:")
        print("  1. Set GROQ_API_KEY if not already set")
        print("  2. Run: ./start_vibevox.sh")
        print("  3. Open: http://localhost:8000/")
        return 0
    else:
        print("⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
