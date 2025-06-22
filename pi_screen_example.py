#!/usr/bin/env python3
"""
Pi Screen Standby Visual Integration Example
Shows how to integrate the mystical floating face standby visual with the AI Pillar hardware controller.
"""

import time
import logging
from hardware_controller import HardwareController, HardwareConfig, AIState, start_standby_visual, stop_standby_visual
from pi_screen_standby import PiScreenStandby

def example_standalone_usage():
    """Example of using the standby visual standalone"""
    print("Standalone Pi Screen Standby Visual Example")
    print("=" * 50)
    
    # Create and initialize the standby visual
    standby = PiScreenStandby(screen_width=800, screen_height=480, fps=60)
    
    if not standby.initialize():
        print("❌ Failed to initialize standby visual")
        return False
    
    print("✅ Standby visual initialized")
    print("Starting mystical floating face animation...")
    print("Press Ctrl+C to stop")
    
    try:
        standby.start()
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping standby visual...")
    finally:
        standby.cleanup()
        print("✅ Standby visual stopped")
    
    return True

def example_hardware_integration():
    """Example of integrating with the hardware controller"""
    print("\nHardware Controller Integration Example")
    print("=" * 50)
    
    # Create hardware configuration
    config = HardwareConfig()
    
    # Initialize hardware controller
    controller = HardwareController(config)
    
    if not controller.initialize():
        print("❌ Failed to initialize hardware controller")
        return False
    
    print("✅ Hardware controller initialized")
    
    try:
        # Start the standby visual
        print("Starting standby visual...")
        controller.start_standby_visual()
        
        # Simulate different AI states
        print("Simulating AI states...")
        
        for state in [AIState.STARTUP, AIState.IDLE, AIState.THINKING, AIState.LISTENING, AIState.SPEAKING]:
            print(f"Setting AI state: {state.value}")
            controller.set_ai_state(state)
            time.sleep(3)
        
        # Keep the standby visual running
        print("Standby visual is running in background...")
        print("Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping hardware controller...")
    finally:
        controller.cleanup()
        print("✅ Hardware controller stopped")
    
    return True

def example_global_functions():
    """Example using global functions"""
    print("\nGlobal Functions Example")
    print("=" * 50)
    
    # Initialize hardware (this will also initialize the Pi screen)
    from hardware_controller import initialize_hardware
    
    if not initialize_hardware():
        print("❌ Failed to initialize hardware")
        return False
    
    print("✅ Hardware initialized")
    
    try:
        # Start standby visual using global function
        print("Starting standby visual using global function...")
        start_standby_visual()
        
        # Simulate some activity
        print("Standby visual is running...")
        print("Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping standby visual...")
    finally:
        # Clean up using global function
        from hardware_controller import cleanup_hardware
        cleanup_hardware()
        print("✅ Hardware cleaned up")
    
    return True

def main():
    """Main example function"""
    print("Pi Screen Standby Visual Examples")
    print("=" * 60)
    print("This demonstrates different ways to use the mystical floating face")
    print("standby visual for Raspberry Pi 4 touchscreen.")
    print()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example 1: Standalone usage
    print("1. Standalone Usage Example")
    print("-" * 30)
    example_standalone_usage()
    
    # Example 2: Hardware integration
    print("\n2. Hardware Controller Integration Example")
    print("-" * 30)
    example_hardware_integration()
    
    # Example 3: Global functions
    print("\n3. Global Functions Example")
    print("-" * 30)
    example_global_functions()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("\nKey Features:")
    print("- Mystical floating face with gentle floating motion")
    print("- Animated particle background (mist effect)")
    print("- Realistic eye blinking animation")
    print("- Rotating aura lines with varying intensity")
    print("- Touch event handling for interaction")
    print("- Smooth 60 FPS animation")
    print("- Integration with existing hardware controller")
    print("- Proper resource cleanup and management")

if __name__ == "__main__":
    main() 