#!/usr/bin/env python3
"""
Test script for Raspberry Pi 4 AI Pillar Hardware Integration
Tests OLED display, RGB ring, and LED strip functionality
"""

import asyncio
import logging
import time
from hardware_controller import HardwareController, HardwareConfig, AIState
from ai_pillar_integration import PillarConfig, PillarMode, initialize_pillar, shutdown_pillar

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_hardware_components():
    """Test individual hardware components"""
    logger.info("Testing hardware components...")
    
    # Initialize hardware controller
    config = HardwareConfig()
    controller = HardwareController(config)
    
    if not controller.initialize():
        logger.error("Failed to initialize hardware controller")
        return False
    
    try:
        # Test OLED display
        logger.info("Testing OLED display...")
        controller.oled.show_text("OLED Test")
        await asyncio.sleep(2)
        
        # Test RGB ring
        logger.info("Testing RGB ring...")
        controller.rgb_ring.set_color((255, 0, 0))  # Red
        await asyncio.sleep(1)
        controller.rgb_ring.set_color((0, 255, 0))  # Green
        await asyncio.sleep(1)
        controller.rgb_ring.set_color((0, 0, 255))  # Blue
        await asyncio.sleep(1)
        
        # Test LED strip
        logger.info("Testing LED strip...")
        controller.led_strip.thinking_animation()
        await asyncio.sleep(3)
        controller.led_strip.stop_animation()
        
        logger.info("Hardware component tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Hardware test failed: {e}")
        return False
    finally:
        controller.cleanup()

async def test_ai_states():
    """Test AI state transitions with visual feedback"""
    logger.info("Testing AI state transitions...")
    
    # Initialize pillar integration
    config = PillarConfig(
        mode=PillarMode.STANDALONE,
        enable_voice=False,  # Disable voice for testing
        enable_visual_feedback=True
    )
    
    if not await initialize_pillar(config):
        logger.error("Failed to initialize AI Pillar")
        return False
    
    try:
        # Test each AI state
        states = [
            AIState.STARTUP,
            AIState.IDLE,
            AIState.THINKING,
            AIState.SPEAKING,
            AIState.LISTENING,
            AIState.ERROR,
            AIState.IDLE
        ]
        
        for state in states:
            logger.info(f"Testing state: {state.value}")
            from ai_pillar_integration import get_pillar_integration
            pillar = await get_pillar_integration()
            await pillar.set_ai_state(state)
            await asyncio.sleep(2)
        
        logger.info("AI state tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"AI state test failed: {e}")
        return False
    finally:
        await shutdown_pillar()

async def test_message_processing():
    """Test message processing with visual feedback"""
    logger.info("Testing message processing...")
    
    # Initialize pillar integration
    config = PillarConfig(
        mode=PillarMode.STANDALONE,
        enable_voice=False,
        enable_visual_feedback=True
    )
    
    if not await initialize_pillar(config):
        logger.error("Failed to initialize AI Pillar")
        return False
    
    try:
        # Test message processing
        test_messages = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke"
        ]
        
        from ai_pillar_integration import get_pillar_integration
        pillar = await get_pillar_integration()
        
        for message in test_messages:
            logger.info(f"Processing message: {message}")
            response = await pillar.process_message(message)
            logger.info(f"Response: {response.get('response', 'No response')}")
            await asyncio.sleep(1)
        
        logger.info("Message processing tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Message processing test failed: {e}")
        return False
    finally:
        await shutdown_pillar()

async def test_simulation_mode():
    """Test running in simulation mode without hardware"""
    logger.info("Testing simulation mode...")
    
    # Initialize pillar integration in simulation mode
    config = PillarConfig(
        mode=PillarMode.SIMULATION,
        enable_voice=False,
        enable_visual_feedback=True
    )
    
    if not await initialize_pillar(config):
        logger.error("Failed to initialize AI Pillar in simulation mode")
        return False
    
    try:
        # Test message processing in simulation
        from ai_pillar_integration import get_pillar_integration
        pillar = await get_pillar_integration()
        
        response = await pillar.process_message("Hello from simulation mode")
        logger.info(f"Simulation response: {response.get('response', 'No response')}")
        
        logger.info("Simulation mode tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Simulation mode test failed: {e}")
        return False
    finally:
        await shutdown_pillar()

async def main():
    """Run all hardware tests"""
    logger.info("Starting Raspberry Pi 4 AI Pillar Hardware Tests")
    
    tests = [
        ("Hardware Components", test_hardware_components),
        ("AI State Transitions", test_ai_states),
        ("Message Processing", test_message_processing),
        ("Simulation Mode", test_simulation_mode)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Hardware integration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check hardware connections and dependencies.")

if __name__ == "__main__":
    asyncio.run(main()) 