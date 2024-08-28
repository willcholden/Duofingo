# Duofingo - Language learning for your hands

![main menu](https://github.com/willcholden/Duofingo/blob/main/logo.png)

## Introduction

The language-learning platform Duolingo has over 100 million monthly active users who can select any of their 40 offered languages. The game-like style of learning works great for most languages, but what if a user wants to learn a language that doesn't use text or audio? Currently Duolingo does not offer any lessons for American Sign Language (ASL). Enter Duofingo- a revolutionary tool for users to practice their ASL skills at their own pace with just a computer. 

![example gameplay](https://github.com/willcholden/Duofingo/blob/main/example_gameplay.png)

## Gameplay

The rules are simple, just sign as many correct terms as you can in 60 seconds. Once the timer begins, your target word will appear at the bottom left corner of the screen ("butter" in the example above). Once you correctly sign a word, your score will increment by one point and the target word will change. Your current score can be seen at the bottom center of the screen, and the countdown timer is visible in the bottom right corner. If your final score is greater than the current high score for a category, the high score will be replaced. 

![end title](https://github.com/willcholden/Duofingo/blob/main/end_title.png)

## Game Engine

The two main python packages that allowed me to create this game are OpenCV and Mediapipe. 

OpenCV is a software library that lets developers create applications using a computer vision infrastructure. Mediapipe is a package that uses artificial intelligence within a computer vision framework, allowing users to create projects where the computer camera can "detect" certain objects. I used mediapipe to detect the players' hands and generate data from the positions of their fingers and joints (called "landmarks"). 

![mediapipe landmarks](https://github.com/willcholden/Duofingo/blob/main/mediapipe_landmarks.png)

For each of the five playable categories (letters, animals, foods, verbs, animals), I created training data by signing all of the items and saving the landmark information to a dictionary. Once the landmark data was saved, I used it to train random forest classifiers so that they could predict which sign a user is displaying. Once the game begins, it randomly takes words from the category's item list to use as the target. The user must then sign so that the prediction output matches the target. While this method of classification was eventually successful, it came with a host of complications. 

## Challenges

One major challenge I had to overcome was how to handle signs that required movement. For example, to sign the word "hello" you begin by placing your hand near your forehead, then move it away. Since the predictor can only make predictions from static data, it has no way of knowing what direction(s) your hands are moving. To combat this I introduced the concept of memory in the form of a prediction queue. The prediction queue stores the current prediction as well as the four previous predictions. Using this idea of prediction memory, I deconstructed signs with movement into multiple parts (e.g. "hand1" and "hand2"). With this framework, I programmed the game to check the prediction queue any time a secondary movement sign was detected. For example, if the predictor outputs "hand2" the game will check the prediction queue for "hand1". Then, and only then, the player will receive credit for signing the correct word. 

Another complication I faced was handedness; whether the user was showing their left hand, their right hand, no hands, or both hands. All of the predictors expected 84 data points- each of the 21 landmarks per hand is composed of an x-component and a y-component, so 21 landmarks * two components * two hands = 84 total data points. However, mediapipe will only generate data for hands that the computer camera can see. So, if a user shows only one hand, the prediction data will contain only 42 points, which causes an error. To resolve this I decided to "pad" single-handed data with 0's (42 0's at the beginning of the list for right handedness, or 42 0's at the end of the list for left handedness). Then, every mediapipe output would contain exactly 84 points. If a user shows no hands, the game will not even attempt to predict, since all of the signs I included require at least one hand. 

Lastly, an issue I faced while creating the "letter" category was similarity between signs. Since there are 26 letters, the classifier would constantly make mistakes even if the user was signing correctly. Either the predictor would give credit for an incorrect answer, or it would fail to give credit for a correct answer. I solved this problem via a two-pronged approach. First, I introduced the red square. As you can see, there are red squares visible on the screen during gameplay. I trained this category while keeping my palm directly in the center of the square, so that users' attempts will be based solely on the relative positions of their fingers and not where their hands deviate off to. 

![letter example](https://github.com/willcholden/Duofingo/blob/main/letter_example.png)

The second solution I introduced for this problem was the random forest "predict_proba" method. A user would receive credit for a correct sign only if the predictor was at least 35% confident that the user was indeed correct. This is helpful for similar signs like "u" and "r" where the only difference is a slight shift between the index and middle fingers. With these two methods in place, the predictor became extramely accurate when deciding to give credit or not. 

## Video Demonstrations

### Letters

[![Letters Demo](https://img.youtube.com/vi/q5lIiUTLBZU/0.jpg)](https://www.youtube.com/watch?v=q5lIiUTLBZU&t=1m30s)


[Letters Demo at 1:30](https://www.youtube.com/watch?v=q5lIiUTLBZU&t=1m30s)
