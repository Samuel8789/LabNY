
#include "QuickStats.h" // quickstats is used for computing quick statistics
// Arduino script for an arduino-based dual lick detector that works with piezosensors
// MOL 2017: Made script
// Left = 1, Right = 2;

QuickStats stats; // Instance of the QuickStats library

//Variables for the sensorvalues:
const int   SENSORPIN1 = A2; // Sensor input left
const int   SENSORPIN2 = A3; // Sensor input right
float       sensorValue1; // value at sensor 1
float       sensorValue2; // value at sensor 2
float       thres = 0.5; // threshold for sensor lick detection (in Volts)
boolean     bSuccRight = 1;
boolean     bSuccLeft = 1;
unsigned long lastlick = 0;

// Variables for the baseline
float       baseline1[40]; // vector containing baseline values for sensor 1
float       baseline2[40]; // vector containing baseline values for sensor 2
int         baselineSamplePeriod = 50; // sampling period for baseline in ms
float       medBase1 = 0; // median of baseline sensor 1
float       medBase2 = 0; // median of baseline sensor 2

//Variables for the valves
int         portswitch  = 13; // switch sides pin for reward ports
int         portbrake   = 8; // brake pin for reward ports
int         portpower   = 11; // on off pin for reward ports
int         portswitch2   = 12; // brake pin for reward ports
int         portbrake2   = 9; // brake pin for reward ports
int         rewardTime1 = 200; // Initialize value reward time for port 1 i.e. right side
int         rewardTime2 = 200; // Initialize value reward time for port 2 i.e. left side
unsigned long tvalveopen = 0; // time of opening valve

//Variables for communication:
unsigned long baudrate = 115200; // baudrate, should be the same as reader (i.e. matlab)
char        Ctrl = 'A'; // Var for storing serial connection character
int         ttl_dur = 10; //Duration of a TTL pulse in ms

//Neuralynx TTL table (Bits 4-7 are used:)
#define TTL_leftLick      B00010000
#define TTL_rightLick     B00100000
#define TTL_leftReward    B00110000
#define TTL_rightReward   B01000000

//Other available codes:
//B01010000
//B01100000
//B10000000
//B10010000
//B10100000


void setup()
{
  Serial.begin(baudrate);

  //establish correct valve output settings:
  pinMode(portswitch2, OUTPUT); //CH A -- HIGH = forwards and LOW = backwards???
  pinMode(portswitch, OUTPUT); //CH B -- HIGH = forwards and LOW = backwards???
  //establish motor brake pins
  pinMode(portbrake2, OUTPUT); //brake (disable) CH A
  pinMode(portbrake, OUTPUT); //brake (disable) CH B
  digitalWrite(portbrake2, HIGH);  //DISABLE CH A
  digitalWrite(portbrake, HIGH); // DISABLE CH B
  digitalWrite(portswitch2, HIGH);   //Sets direction of CH A (HIGH is positive)
  digitalWrite(portswitch, HIGH);   //Sets direction of CH B (HIGH is positive)
  analogWrite(portpower, 255);   //Turns on power for driving the ports

  analogReference(DEFAULT);

  //Initialize ouput port to communicate with Neuralynx/devices (digital)
  DDRD = DDRD | B11110000; //Set port D pins 4-7 as output (0/1 not usable, 3 PWM A for motor, so uses 4-7)
  PORTD = B00000000; //Set port D to 0

  // Determine baseline:
  for (int i = 0; i < 40; i++) {
    analogRead(SENSORPIN1);
    baseline1[i] = analogRead(SENSORPIN1) / 1024.0 * 5.0;
    analogRead(SENSORPIN2);
    baseline2[i] = analogRead(SENSORPIN2) / 1024.0 * 5.0;
    delay(baselineSamplePeriod);
  }

  // Get the median of the baseline
  medBase1 = stats.median(baseline1, 40);
  medBase2 = stats.median(baseline2, 40);

}

// This is the actual loop that the arduino continuously runs through
void loop()
{
  checkSensors();

} // end loop

// Additional functions:

// serialEvent is run every loop whenever there is input from the serial connection
// If this is the case, check the serial connection
void serialEvent()
{
  checkSerial();
}

// Check the serial connection for commands
void checkSerial() {
  if (Serial.find("I")) {
    delay(1);
    // Read in the serial message
    Ctrl = Serial.read();
    switch ( Ctrl ) {
      case 'P':
        Ctrl = Serial.read();
        // If right side reward is requested open valve 1
        Serial.println('D'); // send a 'D' to say it worked
        if (Ctrl == 'L') {
          rewardTime1 = Serial.parseInt();
          openOne(1);
        }
        // If left side reward is requested open valve 2
        else if (Ctrl == 'R') {
          rewardTime2 = Serial.parseInt();
          openTwo(1);
        }
        break;

      // Set the threshold for lick detection to value
      case 'F':
        thres = Serial.parseInt();
        thres = thres / 1000;
        thres = thres;
        Serial.println('D');
        break;
      // Send current runtime over serial port
      case 'C':
        Serial.println('R');
        Serial.println(millis());
        break;
      default:
        ;
    }
  }
}

// Check the sensors for having crossed the threshold
void checkSensors() {
  analogRead(SENSORPIN1);
  sensorValue1 = (analogRead(SENSORPIN1) / 1024.0 * 5.0) - medBase1;
  sensorValue1 = abs(sensorValue1);
  analogRead(SENSORPIN2);
  sensorValue2 = (analogRead(SENSORPIN2) / 1024.0 * 5.0) - medBase2;
  sensorValue2 = abs(sensorValue2);

  ///Optional part to plot:
//  Serial.print(sensorValue1); // Print the voltage.
//  Serial.print(",");          //seperator
//  Serial.print(sensorValue2); // Print the voltage.
//  Serial.print(",");          //seperator
//  Serial.println(thres);      // Print the threshold.
//  Serial.print(",");          //seperator

  // Check if sensor 1 crossed threshold
  if (sensorValue1 > thres) {
    if (bSuccRight) {
      cheetahTTL(TTL_leftLick);
      bSuccRight = false;
      lastlick = millis();
      printLeft();
//      openOne(1);
    }
  }
  else { //reset again if value goes below threshold
    bSuccRight = true;
  }

  // Check if sensor 2 crossed threshold
  if (sensorValue2 > thres) {
    if (bSuccLeft) {
      cheetahTTL(TTL_rightLick);
      bSuccLeft = false;
      lastlick = millis();
      printRight();
//      openTwo(1);
    }
  }
  else { //reset again if value goes below threshold
    bSuccLeft = true;
  }
}

// Open the left valve:
void openOne(int half) { //Left
  tvalveopen = millis();
  cheetahTTL(TTL_leftReward);
  digitalWrite(portswitch, LOW);    //Sets direction of CH B (HIGH is right, LOW is left)
  digitalWrite(portbrake, LOW);     //Allows 24V over motor=valve
  //  delay(rewardTime1);
  while (millis() - tvalveopen < rewardTime1) {
    checkSensors();
  }
  digitalWrite(portbrake, HIGH);    //Allows 24V over motor=valve
}

// Open the right valve:
void openTwo(int half) { //Right
  tvalveopen = millis();
  cheetahTTL(TTL_rightReward);
  digitalWrite(portswitch, HIGH);   //Sets direction of CH B (HIGH is right)
  digitalWrite(portbrake, LOW);     //Allows 24V over motor=valve
  //  delay(rewardTime2);
  while (millis() - tvalveopen < rewardTime2) {
    checkSensors();
  }
  digitalWrite(portbrake, HIGH);    //Allows 24V over motor=valve
}

// Print occurence of RIGHT lick to Serial Port
void printRight() {
  Serial.println('O');
  Serial.println("Y");
  Serial.println(lastlick);
}

// Print occurence of LEFT lick to Serial Port
void printLeft() {
  Serial.println('O');
  Serial.println("Z");
  Serial.println(lastlick);
}

void cheetahTTL(byte ttl_code) {
  PORTD = ttl_code;
  delay(ttl_dur);
  PORTD = B00000000;
}



