#!/usr/bin/env python3
"""
Test script for Pi Screen Standby Visual
Demonstrates the mystical floating face standby visual for Raspberry Pi 4 touchscreen.
"""

import time
import logging
from pi_screen_standby import PiScreenStandby, run_standby_visual

def test_standby_visual():
    """Test the standby visual functionality"""
    print("Testing Pi Screen Standby Visual")
    print("=" * 40)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test 1: Create and initialize standby visual
    print("\n1. Creating standby visual instance...")
    standby = PiScreenStandby(screen_width=800, screen_height=480, fps=60)
    
    print("2. Initializing standby visual...")
    if not standby.initialize():
        print("❌ Failed to initialize standby visual")
        return False
    
    print("✅ Standby visual initialized successfully")
    
    # Test 2: Start the visual (non-blocking)
    print("\n3. Starting standby visual (will run for 10 seconds)...")
    if not standby.start():
        print("❌ Failed to start standby visual")
        standby.cleanup()
        return False
    
    print("✅ Standby visual started successfully")
    print("   - Touch the screen to see touch events")
    print("   - Press ESC key to exit early")
    
    # Let it run for 10 seconds
    try:
        for i in range(10, 0, -1):
            print(f"   Running for {i} more seconds...", end='\r')
            time.sleep(1)
        print("\n")
    except KeyboardInterrupt:
        print("\n   Interrupted by user")
    
    # Test 3: Stop and cleanup
    print("\n4. Stopping and cleaning up...")
    standby.stop()
    standby.cleanup()
    print("✅ Standby visual stopped and cleaned up successfully")
    
    return True

def test_blocking_run():
    """Test the blocking run function"""
    print("\n" + "=" * 40)
    print("Testing Blocking Run Function")
    print("=" * 40)
    
    print("This will run the standby visual in blocking mode.")
    print("Press Ctrl+C to stop or ESC key to exit.")
    
    try:
        success = run_standby_visual(screen_width=800, screen_height=480, fps=60)
        if success:
            print("✅ Blocking run completed successfully")
        else:
            print("❌ Blocking run failed")
        return success
    except KeyboardInterrupt:
        print("\n✅ Blocking run interrupted by user")
        return True

def main():
    """Main test function"""
    print("Pi Screen Standby Visual Test")
    print("=" * 50)
    print("This test demonstrates the mystical floating face standby visual")
    print("for Raspberry Pi 4 touchscreen with particle effects and animations.")
    print()
    
    # Test 1: Non-blocking mode
    if not test_standby_visual():
        print("\n❌ Non-blocking test failed")
        return
    
    # Ask user if they want to test blocking mode
    print("\n" + "=" * 50)
    response = input("Do you want to test the blocking run mode? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        test_blocking_run()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nFeatures demonstrated:")
    print("- Mystical floating face with gentle motion")
    print("- Animated particle background (mist effect)")
    print("- Blinking eyes with realistic timing")
    print("- Rotating aura lines around the face")
    print("- Touch event handling")
    print("- Smooth 60 FPS animation")
    print("- Proper cleanup and resource management")

if __name__ == "__main__":
    main() 