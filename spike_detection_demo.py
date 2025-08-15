# spike_detection_demo.py - Demo & Test Script for Visual Audio Spike Detection
# ===============================================================================

import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from fast_golf_processor import EnhancedGolfProcessor, VisualAudioSpikeDetector
    print("✅ Enhanced processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import enhanced processor: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpikeDetectionDemo:
    """Demo class to test and visualize spike detection"""
    
    def __init__(self):
        self.processor = EnhancedGolfProcessor()
        self.spike_detector = VisualAudioSpikeDetector()
        print("🚀 Spike Detection Demo initialized")
    
    def test_with_sample_video(self, video_path: str, output_dir: str = "demo_output"):
        """
        Test spike detection with a sample video and create visualizations
        """
        print(f"\n🎬 Testing spike detection with: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: Extract audio waveform
            print("🎵 Extracting audio waveform...")
            audio_array, duration = self.spike_detector.extract_audio_waveform(video_path)
            
            if len(audio_array) == 0:
                print("❌ Failed to extract audio")
                return False
            
            print(f"✅ Audio extracted: {len(audio_array)} samples, {duration:.1f}s duration")
            
            # Step 2: Detect spikes
            print("🔍 Analyzing audio for golf shot spikes...")
            spikes = self.spike_detector.analyze_audio_for_spikes(audio_array, duration)
            
            if not spikes:
                print("❌ No spikes detected")
                return False
            
            print(f"🎯 Detected {len(spikes)} golf shot spikes:")
            for i, spike in enumerate(spikes):
                print(f"   Player {spike.player_id + 1}: {spike.timestamp:.2f}s (confidence: {spike.confidence:.2f})")
            
            # Step 3: Create visualization
            print("📊 Creating spike detection visualization...")
            viz_path = self.spike_detector.visualize_spike_detection(
                audio_array, spikes, 
                os.path.join(output_dir, f"spike_analysis_{os.path.basename(video_path)}.png")
            )
            
            if viz_path:
                print(f"✅ Visualization saved: {viz_path}")
            
            # Step 4: Test full processing pipeline
            print("🎬 Testing complete processing pipeline...")
            
            def demo_progress_callback(progress, message):
                print(f"   📊 {progress}% - {message}")
            
            output_files = self.processor.process_all_videos(
                tee_path=video_path,
                output_dir=output_dir,
                progress_callback=demo_progress_callback
            )
            
            if output_files:
                print(f"🎉 Success! Created {len(output_files)} highlight videos:")
                for filename in output_files:
                    print(f"   ✅ {filename}")
            else:
                print("⚠️ No output files created")
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            return False
    
    def create_synthetic_test_data(self, output_path: str = "demo_output/synthetic_test.wav"):
        """
        Create synthetic audio with golf shot-like spikes for testing
        """
        print("🔧 Creating synthetic test data...")
        
        try:
            import soundfile as sf
            
            # Create synthetic audio data
            duration = 120  # 2 minutes
            sample_rate = 44100
            samples = int(duration * sample_rate)
            
            # Generate background noise
            background = np.random.normal(0, 0.01, samples)
            
            # Add golf shot spikes at specific times
            spike_times = [15, 30, 45, 75, 90, 105]  # 6 shots
            spike_amplitudes = [0.8, 0.7, 0.9, 0.6, 0.85, 0.75]
            
            for i, (spike_time, amplitude) in enumerate(zip(spike_times, spike_amplitudes)):
                spike_start = int(spike_time * sample_rate)
                spike_duration = int(0.1 * sample_rate)  # 100ms spikes
                
                # Create golf shot-like spike (sharp attack, quick decay)
                spike_envelope = np.exp(-np.linspace(0, 10, spike_duration))
                spike_signal = amplitude * spike_envelope * np.sin(2 * np.pi * 2000 * np.linspace(0, 0.1, spike_duration))
                
                # Add spike to background
                end_idx = min(spike_start + spike_duration, len(background))
                background[spike_start:end_idx] += spike_signal[:end_idx - spike_start]
            
            # Save synthetic audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, background, sample_rate)
            
            print(f"✅ Synthetic test data created: {output_path}")
            print(f"   📊 Duration: {duration}s, Sample rate: {sample_rate}Hz")
            print(f"   🎯 Golf shots at: {spike_times} seconds")
            
            return output_path
            
        except ImportError:
            print("⚠️ soundfile not available, skipping synthetic data creation")
            return None
        except Exception as e:
            print(f"❌ Failed to create synthetic data: {e}")
            return None
    
    def analyze_detection_accuracy(self, video_path: str, expected_shot_times: list):
        """
        Analyze the accuracy of spike detection against known shot times
        """
        print(f"\n📊 Analyzing detection accuracy for: {os.path.basename(video_path)}")
        print(f"   Expected shots at: {expected_shot_times}")
        
        try:
            # Extract and analyze
            audio_array, duration = self.spike_detector.extract_audio_waveform(video_path)
            spikes = self.spike_detector.analyze_audio_for_spikes(audio_array, duration)
            
            detected_times = [spike.timestamp for spike in spikes]
            print(f"   Detected shots at: {[f'{t:.1f}s' for t in detected_times]}")
            
            # Calculate accuracy metrics
            tolerance = 5.0  # 5 second tolerance
            matches = 0
            
            for expected_time in expected_shot_times:
                for detected_time in detected_times:
                    if abs(expected_time - detected_time) <= tolerance:
                        matches += 1
                        break
            
            accuracy = matches / len(expected_shot_times) if expected_shot_times else 0
            precision = matches / len(detected_times) if detected_times else 0
            
            print(f"   📈 Results:")
            print(f"      Accuracy: {accuracy:.1%} ({matches}/{len(expected_shot_times)} shots found)")
            print(f"      Precision: {precision:.1%} ({matches}/{len(detected_times)} detections correct)")
            print(f"      False positives: {len(detected_times) - matches}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'matches': matches,
                'false_positives': len(detected_times) - matches,
                'detected_times': detected_times,
                'expected_times': expected_shot_times
            }
            
        except Exception as e:
            print(f"❌ Accuracy analysis failed: {e}")
            return None

def main():
    """Main demo function"""
    print("🏌️ Visual Audio Spike Detection Demo")
    print("=" * 50)
    
    demo = SpikeDetectionDemo()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Sample Tee Shot Video',
            'path': 'sample_videos/tee_shot.mp4',
            'expected_times': [12, 28, 45, 62]  # Example expected shot times
        },
        {
            'name': 'Test Golf Video',
            'path': 'test_video.mp4',
            'expected_times': []  # Unknown timing
        }
    ]
    
    # Run tests
    for scenario in test_scenarios:
        print(f"\n🎯 Testing Scenario: {scenario['name']}")
        print("-" * 30)
        
        if os.path.exists(scenario['path']):
            # Test spike detection
            success = demo.test_with_sample_video(scenario['path'])
            
            # Analyze accuracy if expected times provided
            if success and scenario['expected_times']:
                demo.analyze_detection_accuracy(scenario['path'], scenario['expected_times'])
        else:
            print(f"⚠️ Video file not found: {scenario['path']}")
            print("   Please place a test video file at this location to run the demo")
    
    # Create synthetic test if possible
    print(f"\n🔧 Synthetic Test Data")
    print("-" * 30)
    synthetic_path = demo.create_synthetic_test_data()
    
    if synthetic_path:
        print("ℹ️ You can now test the spike detector with the synthetic audio file")
        print("   This file contains 6 artificial golf shots with known timing")
    
    print(f"\n✅ Demo completed!")
    print("📁 Check the 'demo_output' directory for visualization files")
    
    # Instructions for users
    print(f"\n📋 To test with your own videos:")
    print("   1. Place your tee shot video in the current directory")
    print("   2. Update the test_scenarios list with your video path")
    print("   3. Run this script again")
    print("   4. Check the generated visualization images")

def quick_test(video_path: str):
    """Quick test function for command line usage"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🚀 Quick spike detection test for: {os.path.basename(video_path)}")
    
    demo = SpikeDetectionDemo()
    success = demo.test_with_sample_video(video_path, "quick_test_output")
    
    if success:
        print("✅ Quick test completed successfully!")
    else:
        print("❌ Quick test failed")

if __name__ == "__main__":
    # Command line usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        quick_test(video_path)
    else:
        main()