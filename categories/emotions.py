#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Emotion(Oi):
    # FER-2013
    Oi.FACE.ANGRY = Name('Angry')
    Oi.FACE.DISGUST = Name('Disgust')
    Oi.FACE.FEAR = Name('Fear')
    Oi.FACE.HAPPY = Name('Happy')
    Oi.FACE.NEUTRAL = Name('Neutral')
    Oi.FACE.SAD = Name('Sad')
    Oi.FACE.SURPRISE = Name('Surprise')
