/* MEng Capstone Project Codebase
 *  Spring 2021
 *  Karl Mueller, Masters Candiate in Mechanical Engineering
 * _________________________________________________________
 * 
 * Edge Detection Scanning method for 3D interpretation of human arm
 */

 #include <AccelStepper.h>

 #define dirPin 5   
 #define stepPin 6
 #define motorInterfaceType 1

 #define m0pin 10 
 #define m1pin 9
 #define m2pin 8

 #define homeLimitPin 13 //pin position of homing limit switch

 //define interrupt and control pins for the encoder
 
//Pin definition

const int nextMode = 2; //clock pin for the encoder, iterrupt pin
const int encoderDT = 4; //data pin of the encoder
const int homePin = 3; //

//Variable and counting def
int stepsPerRev;
int buttonCounter;
int rotationCounter;

int clkNow;
int clkPrev;

int dtNow;
int dtPrev;

float timeNow1; //for debouncing of button? : https://www.youtube.com/watch?v=YGnxGJmxHlo
float timeNow2;
float debounceDelay = 750; //time in ms used to debounce button input, prevents input for XX ms

long homing;
long back_off;

long homePos;
long point1;

long nextPos;

AccelStepper stepper = AccelStepper(motorInterfaceType, stepPin, dirPin);
 
void setup() {
  Serial.begin(9600);


    pinMode(nextMode, INPUT_PULLUP);
    pinMode(homePin, INPUT_PULLUP);
    
    pinMode(m0pin, OUTPUT);
    pinMode(m1pin, OUTPUT);
    pinMode(m2pin, OUTPUT);

    pinMode(homeLimitPin, INPUT_PULLUP);

    digitalWrite(m0pin, LOW);
    digitalWrite(m1pin, HIGH);
    digitalWrite(m2pin, HIGH);


    //clkPrev = digitalRead(encoderCLK);
    //dtPrev = digitalRead(encoderDT);

    attachInterrupt(digitalPinToInterrupt(homePin), setGoalToHome, FALLING);
    attachInterrupt(digitalPinToInterrupt(nextMode), setGoalToNext, FALLING);

    stepper.setMaxSpeed(10000); //steps/second
    stepper.setAcceleration(10000); //steps/second**2

    timeNow1 = millis();
    
    Serial.print("stepper is homing...");
    home();
    stepper.stop();
    homePos = stepper.currentPosition();
    Serial.print(homePos);

    stepper.enableOutputs();
    stepper.setSpeed(-6000); //set new speed
    stepper.runToNewPosition(homePos - 2000); //move from home to point 1
    point1 = stepper.currentPosition();
    //Serial.print(stepper.currentPosition());
  
}

void loop() {

  runMotor();
  /*
  stepper.setSpeed(-4000); //set new speed
  delay(1000);

  stepper.runToNewPosition(point1 - 2756); //move from home to point 1

  delay(1000);
  
  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);
  
  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(stepper.currentPosition() - 2756); //move from home to point 1

  delay(1000);

  stepper.runToNewPosition(point1 + 500    ); //move from end to past home

  delay(750);

  stepper.runToNewPosition(point1); //move to home from beyond to remove backlash
  */

}

void setGoalToHome() {
  timeNow2 = millis();
  if(timeNow2 >= timeNow1 + debounceDelay) { 
    Serial.println("goal is set to home");
    //stepper.enableOutputs();
    //stepper.setSpeed(4000);
    //stepper.runToNewPosition(point1 + 500    ); //move from end to past home
    //delay(750);
    //stepper.runToNewPosition(point1); //move to home from beyond to remove backlash
    nextPos = point1+500;
    
    timeNow1 = millis();
  }
}

void home() {
    //initialize the program via homing of the device
  stepper.setSpeed(7500);
  while(digitalRead(homeLimitPin)){
    stepper.moveTo(homing);
    homing++;
    stepper.run();
    //Serial.print(stepper.currentPosition());
    delay(1);
  }
  back_off = homing-1;
  stepper.setSpeed(250);
  while(!digitalRead(homeLimitPin)){

    stepper.moveTo(back_off);
    back_off--;
    stepper.run();
    //Serial.print(stepper.currentPosition());
    delay(1);
  }
}


void setGoalToNext(){
  timeNow2 = millis();
  if(timeNow2 >= timeNow1 + debounceDelay) { 
    Serial.println("goal is set to next!");
    //stepper.enableOutputs();
    //stepper.setSpeed(-4000);
    //stepper.runToNewPosition(stepper.currentPosition() - 2756);
    //stepper.run();
    nextPos = stepper.currentPosition() - 2756;

    
    timeNow1 = millis();
  }
  
  
  /*clkNow = digitalRead(encoderCLK); //reads clock state of pin

  if(clkNow != clkPrev && clkNow ==1){

    if(digitalRead(encoderDT) != clkNow){

      rotationCounter += 1;
    } else {
      rotationCounter-= 1;
    }
  }
  clkPrev = clkNow;
  */
  delay(500);
}

void runMotor() {
  stepper.setSpeed(-2000);
  stepper.enableOutputs();
  stepper.moveTo(nextPos);

  while(stepper.distanceToGo() != 0) {
    stepper.runToNewPosition(nextPos);
  }

  if(nextPos==point1+500){
    stepper.runToNewPosition(point1+1);
    nextPos = stepper.currentPosition();
  }
}
