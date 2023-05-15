#include <TimerOne.h>

bool Time_Second_has_gone = false;
int seconds = 0;

int communication_seconds = 0;
volatile int Left_Pulses; //Amount of pulses left
int Sensor_Left = 2; // Connection of sensor left

volatile int Right_Pulses; //Amount of pulses right
int Sensor_Right = 3; //Connection of sensor right

void rpmLeft ()  //interupt call left
{
  Left_Pulses++;
 }

void rpmRight ()  //interupt call right
{
  Right_Pulses++;
}

void Timer1Interrupt()
{
  Time_Second_has_gone = true; 
}

// The setup() method runs once, when the sketch starts
void setup() //
{
  Serial.begin(115200); //This is the setup function where the serial port is initialised,
  pinMode(Sensor_Left, INPUT); //initializes digital pin as an input
  pinMode(Sensor_Right, INPUT); //initializes digital pin as an input
  digitalWrite(Sensor_Left, INPUT_PULLUP);
  digitalWrite(Sensor_Right, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(Sensor_Left), rpmLeft, FALLING); //and the interrupt is attached
  attachInterrupt(digitalPinToInterrupt(Sensor_Right), rpmRight, FALLING); //and the interrupt is attached

  Time_Second_has_gone = false;

  Left_Pulses = 0;  //Set to 0 ready for calculations
  Right_Pulses = 0;   //Set  to 0 ready for calculations

  Timer1.initialize(1000000);  // elke seconde interrupt
  Timer1.attachInterrupt(Timer1Interrupt);  // attaches Timer1Interrupt() as a timer overflow interrupt
}

void toon()
{
  Serial.print (seconds, DEC);
  Serial.print (" Links \t");
  Serial.print (Left_Pulses, DEC);
  Serial.print (" \t");
  Serial.print (" Rechts \t");
  Serial.print (Right_Pulses, DEC);
  Serial.print("\r\n");
  Serial.print("\r\n");
}
 
// the loop() method runs over and over again,
// as long as the Arduino has power
void loop ()
{
  interrupts();  //Enables interrupts

  if ( Time_Second_has_gone == true)
  {
    noInterrupts();
    Time_Second_has_gone = false;
    seconds++;
    toon ();
    Right_Pulses = 0;
    Left_Pulses = 0;
  }
}
