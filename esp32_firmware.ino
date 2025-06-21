/*
 * ESP32 Firmware for AI Pillar
 * Communicates with Raspberry Pi and manages sensors
 * 
 * Hardware:
 * - ESP32 WROOM (2.4GHz WiFi/BT)
 * - 0.96" OLED Display (I2C)
 * - RGB LED Ring (WS2812B)
 * - LED Strip (WS2812B)
 * 
 * Pin Assignments:
 * - GPIO 21: LED Strip Data
 * - GPIO 18: RGB Ring Data
 * - GPIO 22: I2C SCL (OLED)
 * - GPIO 21: I2C SDA (OLED)
 * - GPIO 2: Built-in LED (status)
 */

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <FastLED.h>

// WiFi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// WebSocket Configuration
WebSocketsServer webSocket = WebSocketsServer(81);
const int WS_PORT = 81;

// OLED Display Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// RGB Ring Configuration
#define RGB_RING_PIN 18
#define RGB_RING_COUNT 16
CRGB rgbRing[RGB_RING_COUNT];

// LED Strip Configuration
#define LED_STRIP_PIN 21
#define LED_STRIP_COUNT 60
CRGB ledStrip[LED_STRIP_COUNT];

// Built-in LED for status
#define STATUS_LED_PIN 2

// AI States
enum AIState {
  IDLE,
  THINKING,
  SPEAKING,
  LISTENING,
  ERROR,
  STARTUP
};

AIState currentState = STARTUP;

// Animation variables
unsigned long lastAnimationUpdate = 0;
unsigned long animationInterval = 50; // 50ms for smooth animation
int animationStep = 0;

// JSON document for communication
DynamicJsonDocument jsonDoc(1024);

void setup() {
  Serial.begin(115200);
  
  // Initialize status LED
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);
  
  // Initialize I2C for OLED
  Wire.begin(22, 21); // SCL=22, SDA=21
  
  // Initialize OLED display
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0,0);
    display.println(F("AI Pillar ESP32"));
    display.display();
  }
  
  // Initialize RGB Ring
  FastLED.addLeds<WS2812B, RGB_RING_PIN, GRB>(rgbRing, RGB_RING_COUNT);
  FastLED.setBrightness(50);
  clearRGBRing();
  
  // Initialize LED Strip
  FastLED.addLeds<WS2812B, LED_STRIP_PIN, GRB>(ledStrip, LED_STRIP_COUNT);
  FastLED.setBrightness(30);
  clearLEDStrip();
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    toggleStatusLED();
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Start WebSocket server
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  
  // Set startup state
  setAIState(STARTUP);
  
  Serial.println("ESP32 AI Pillar initialized");
}

void loop() {
  webSocket.loop();
  
  // Update animations
  if (millis() - lastAnimationUpdate > animationInterval) {
    updateAnimations();
    lastAnimationUpdate = millis();
  }
  
  // Send status updates periodically
  static unsigned long lastStatusUpdate = 0;
  if (millis() - lastStatusUpdate > 5000) { // Every 5 seconds
    sendStatusUpdate();
    lastStatusUpdate = millis();
  }
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
      
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
        
        // Send initial status
        sendStatusUpdate();
      }
      break;
      
    case WStype_TEXT:
      {
        Serial.printf("[%u] Received: %s\n", num, payload);
        
        // Parse JSON command
        DeserializationError error = deserializeJson(jsonDoc, payload);
        if (error) {
          Serial.println("JSON parsing failed");
          return;
        }
        
        // Process command
        const char* command = jsonDoc["command"];
        if (command) {
          processCommand(command, jsonDoc["data"]);
        }
      }
      break;
  }
}

void processCommand(const char* command, JsonObject data) {
  if (strcmp(command, "set_state") == 0) {
    const char* stateStr = data["state"];
    if (stateStr) {
      if (strcmp(stateStr, "idle") == 0) setAIState(IDLE);
      else if (strcmp(stateStr, "thinking") == 0) setAIState(THINKING);
      else if (strcmp(stateStr, "speaking") == 0) setAIState(SPEAKING);
      else if (strcmp(stateStr, "listening") == 0) setAIState(LISTENING);
      else if (strcmp(stateStr, "error") == 0) setAIState(ERROR);
      else if (strcmp(stateStr, "startup") == 0) setAIState(STARTUP);
    }
  }
  else if (strcmp(command, "set_oled_text") == 0) {
    const char* text = data["text"];
    if (text) {
      displayOLEDText(text);
    }
  }
  else if (strcmp(command, "set_rgb_ring") == 0) {
    int r = data["r"] | 0;
    int g = data["g"] | 0;
    int b = data["b"] | 0;
    setRGBRingColor(r, g, b);
  }
  else if (strcmp(command, "set_led_strip") == 0) {
    int r = data["r"] | 0;
    int g = data["g"] | 0;
    int b = data["b"] | 0;
    setLEDStripColor(r, g, b);
  }
  else if (strcmp(command, "get_status") == 0) {
    sendStatusUpdate();
  }
}

void setAIState(AIState state) {
  currentState = state;
  animationStep = 0;
  
  switch(state) {
    case IDLE:
      displayOLEDText("AI Ready");
      setRGBRingColor(0, 0, 10); // Dim blue
      clearLEDStrip();
      break;
      
    case THINKING:
      displayOLEDText("Thinking...");
      clearRGBRing();
      // LED strip animation will be handled in updateAnimations()
      break;
      
    case SPEAKING:
      displayOLEDText("Speaking");
      // RGB ring pulse animation will be handled in updateAnimations()
      clearLEDStrip();
      break;
      
    case LISTENING:
      displayOLEDText("Listening...");
      // RGB ring pulse animation will be handled in updateAnimations()
      clearLEDStrip();
      break;
      
    case ERROR:
      displayOLEDText("Error");
      setRGBRingColor(255, 0, 0); // Red
      clearLEDStrip();
      break;
      
    case STARTUP:
      displayOLEDText("Starting...");
      // Startup animation will be handled in updateAnimations()
      break;
  }
  
  // Send state change notification
  sendStateChange(state);
}

void updateAnimations() {
  switch(currentState) {
    case THINKING:
      updateThinkingAnimation();
      break;
      
    case SPEAKING:
      updateSpeakingAnimation();
      break;
      
    case LISTENING:
      updateListeningAnimation();
      break;
      
    case STARTUP:
      updateStartupAnimation();
      break;
  }
}

void updateThinkingAnimation() {
  // Top-to-bottom wave animation on LED strip
  clearLEDStrip();
  
  int wavePosition = (animationStep * 2) % (LED_STRIP_COUNT * 2);
  for (int i = 0; i < LED_STRIP_COUNT; i++) {
    int distance = abs(i - wavePosition);
    if (distance < 10) {
      int brightness = 255 - (distance * 25);
      brightness = max(0, brightness);
      ledStrip[i] = CRGB(0, 0, brightness);
    }
  }
  
  FastLED.show();
  animationStep++;
}

void updateSpeakingAnimation() {
  // Pulse animation on RGB ring
  int brightness = 128 + 127 * sin(animationStep * 0.2);
  brightness = max(0, brightness);
  
  for (int i = 0; i < RGB_RING_COUNT; i++) {
    rgbRing[i] = CRGB(0, brightness, 0); // Green
  }
  
  FastLED.show();
  animationStep++;
}

void updateListeningAnimation() {
  // Pulse animation on RGB ring (yellow)
  int brightness = 128 + 127 * sin(animationStep * 0.2);
  brightness = max(0, brightness);
  
  for (int i = 0; i < RGB_RING_COUNT; i++) {
    rgbRing[i] = CRGB(brightness, brightness, 0); // Yellow
  }
  
  FastLED.show();
  animationStep++;
}

void updateStartupAnimation() {
  // Rainbow animation on RGB ring
  for (int i = 0; i < RGB_RING_COUNT; i++) {
    int hue = (animationStep * 10 + i * 20) % 256;
    rgbRing[i] = CHSV(hue, 255, 255);
  }
  
  FastLED.show();
  animationStep++;
  
  // After startup animation, switch to idle
  if (animationStep > 100) {
    setAIState(IDLE);
  }
}

void displayOLEDText(const char* text) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(text);
  display.display();
}

void setRGBRingColor(int r, int g, int b) {
  for (int i = 0; i < RGB_RING_COUNT; i++) {
    rgbRing[i] = CRGB(r, g, b);
  }
  FastLED.show();
}

void clearRGBRing() {
  for (int i = 0; i < RGB_RING_COUNT; i++) {
    rgbRing[i] = CRGB(0, 0, 0);
  }
  FastLED.show();
}

void setLEDStripColor(int r, int g, int b) {
  for (int i = 0; i < LED_STRIP_COUNT; i++) {
    ledStrip[i] = CRGB(r, g, b);
  }
  FastLED.show();
}

void clearLEDStrip() {
  for (int i = 0; i < LED_STRIP_COUNT; i++) {
    ledStrip[i] = CRGB(0, 0, 0);
  }
  FastLED.show();
}

void toggleStatusLED() {
  static bool ledState = false;
  ledState = !ledState;
  digitalWrite(STATUS_LED_PIN, ledState ? HIGH : LOW);
}

void sendStatusUpdate() {
  jsonDoc.clear();
  jsonDoc["type"] = "status";
  jsonDoc["state"] = getStateString(currentState);
  jsonDoc["wifi_connected"] = WiFi.status() == WL_CONNECTED;
  jsonDoc["ip_address"] = WiFi.localIP().toString();
  jsonDoc["uptime"] = millis();
  
  String jsonString;
  serializeJson(jsonDoc, jsonString);
  
  webSocket.broadcastTXT(jsonString);
}

void sendStateChange(AIState state) {
  jsonDoc.clear();
  jsonDoc["type"] = "state_change";
  jsonDoc["state"] = getStateString(state);
  
  String jsonString;
  serializeJson(jsonDoc, jsonString);
  
  webSocket.broadcastTXT(jsonString);
}

const char* getStateString(AIState state) {
  switch(state) {
    case IDLE: return "idle";
    case THINKING: return "thinking";
    case SPEAKING: return "speaking";
    case LISTENING: return "listening";
    case ERROR: return "error";
    case STARTUP: return "startup";
    default: return "unknown";
  }
} 