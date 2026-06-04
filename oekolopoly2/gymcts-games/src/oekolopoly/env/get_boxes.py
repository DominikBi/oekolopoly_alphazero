def get_box1(sanitation):

    # Box 1 Sanitation-Environment 0-5
    if   sanitation  <  2: box1 =  0
    elif sanitation  <  8: box1 = -1
    elif sanitation  < 13: box1 = -2
    elif sanitation  < 18: box1 = -3
    elif sanitation  < 21: box1 = -4
    elif sanitation  < 24: box1 = -5
    elif sanitation  < 26: box1 = -6
    elif sanitation  < 28: box1 = -7
    elif sanitation == 28: box1 = -8
    elif sanitation >= 29: box1 = -9

    return box1


def get_box2(sanitation):
     
    # Box 2 Sanitation-Sanitation 0-0
    if   sanitation  < 22: box2 =  0
    elif sanitation  < 24: box2 = -1
    elif sanitation  < 26: box2 = -2
    elif sanitation == 26: box2 = -3
    elif sanitation == 27: box2 = -4
    elif sanitation == 28: box2 = -5
    elif sanitation >= 29: box2 = -6

    return box2


def get_box3(production):
    
    # Box 3 Production-Production 4-4
    if   production <  7:                     box3 = 0
    elif production < 22 or production >= 29: box3 = 1
    elif production < 29:                     box3 = 2
            
    return box3


def get_box4(production):

    # Box 4 Production-Environment 4-5
    if   production  <  5: box4 =  0
    elif production  <  9: box4 =  1
    elif production  < 13: box4 =  2
    elif production  < 15: box4 =  3
    elif production  < 17: box4 =  4
    elif production  < 19: box4 =  5
    elif production  < 21: box4 =  6
    elif production  < 23: box4 =  7
    elif production == 23: box4 =  8
    elif production == 24: box4 =  9
    elif production == 25: box4 = 10
    elif production == 26: box4 = 12
    elif production == 27: box4 = 14
    elif production == 28: box4 = 18
    elif production >= 29: box4 = 22
            
    return box4


def get_boxC (production):

    # Box C Production-Actionpoints 4-A
    if   production <=  1: boxC = -4
    elif production ==  2: boxC = -3
    elif production ==  3: boxC = -2
    elif production ==  4: boxC = -1
    elif production ==  5: boxC =  0
    elif production  <  8: boxC =  2
    elif production  < 11: boxC =  3
    elif production  < 14: boxC =  4
    elif production  < 17: boxC =  5
    elif production  < 20: boxC =  6
    elif production  < 22: boxC =  7
    elif production  < 24: boxC =  8
    elif production  < 26: boxC =  9
    elif production == 26: boxC = 10
    elif production == 27: boxC = 11
    elif production == 28: boxC = 12
    elif production >= 29: boxC =  0
            
    return boxC


def get_boxV(production):

    # Box V Multiplier Production-Actionpoints 4-A
    if   production <=  1: boxV = -4
    elif production ==  2: boxV = -3
    elif production  <  5: boxV = -2
    elif production  <  7: boxV = -1
    elif production  <  9: boxV =  0
    elif production >=  9: boxV =  1
            
    return boxV


def get_box5(environment):
    
    # Box 5 Environment-Environment 5-5
    if   environment  <  4 or environment == 28 or environment >= 29: box5 =  0
    elif environment  < 14 or environment == 27:                      box5 = -1
    elif environment  < 20 or environment == 26:                      box5 = -2
    elif environment == 24:                                           box5 = -4
    elif environment  < 24 or environment == 25:                      box5 = -3

    return box5


def get_box6(environment):
    
    # Box 6 Environment-Quality of Life 5-2
    if   environment  <  8: box6 =   0
    elif environment  < 11: box6 =  -1
    elif environment  < 15: box6 =  -2
    elif environment  < 18: box6 =  -3
    elif environment  < 20: box6 =  -4
    elif environment  < 22: box6 =  -5
    elif environment == 22: box6 =  -6
    elif environment == 23: box6 =  -7
    elif environment == 24: box6 =  -8
    elif environment == 25: box6 = -10
    elif environment == 26: box6 = -12
    elif environment == 27: box6 = -14
    elif environment == 28: box6 = -18
    elif environment >= 29: box6 = -25

    return box6


def get_box7(education):
    
    # Box 7 Education-Education 1-1
    if   education <  3 or education in range(6, 15) or education >= 29: box7 =  0
    elif education <  6:                                                 box7 = -1
    elif education < 21 or education in range(24, 29):                   box7 =  1
    elif education in range(21, 24):                                     box7 =  2

    return box7


def get_box8(education):

    # Box 8 Education-Quality of Life 1-2
    if   education  <  6: box8 = -2
    elif education  < 10: box8 = -1
    elif education == 10: box8 =  0
    elif education  < 14: box8 =  1
    elif education  < 18: box8 =  2
    elif education  < 22: box8 =  3
    elif education  < 26: box8 =  4
    elif education  < 28: box8 =  5
    elif education >= 28: box8 =  6

    return box8


def get_box9(education, extra_points):
    
    # Box 9 Education-Population Growth 1-3
    if   education ==  1: box9 = 1
    elif education  < 10: box9 = 0
    elif education  < 14: box9 = 1
    elif education  < 19: box9 = 2
    elif education  < 21: box9 = 3
    elif education  < 24: box9 = extra_points
    elif education  < 28: box9 = extra_points
    elif education >= 28: box9 = extra_points

    return box9


def get_box10(quality_of_life):
    
    # Box 10 Quality of Life-Quality of Life 2-2
    if   quality_of_life == 1 or quality_of_life in range( 4, 10) or quality_of_life in range(16, 18) or quality_of_life >= 29: box10 =  0
    elif quality_of_life  < 4 or quality_of_life in range(10, 13) or quality_of_life in range(14, 16):                          box10 =  1
    elif quality_of_life == 13:                                                                                                 box10 =  2
    elif quality_of_life in range(18, 23) or quality_of_life in range(26, 29):                                                  box10 = -1
    elif quality_of_life in range(23, 26):                                                                                      box10 = -2

    return box10


def get_box11(quality_of_life):

    # Box 11 Quality of Life-Population Growth 2-3
    if   quality_of_life <=  1: box11 = -15
    elif quality_of_life ==  2: box11 =  -8
    elif quality_of_life ==  3: box11 =  -6
    elif quality_of_life ==  4: box11 =  -4
    elif quality_of_life ==  5: box11 =  -3
    elif quality_of_life ==  6: box11 =  -2
    elif quality_of_life ==  7: box11 =  -1
    elif quality_of_life ==  8: box11 =   0
    elif quality_of_life ==  9: box11 =   1
    elif quality_of_life  < 13: box11 =   2
    elif quality_of_life  < 21: box11 =   1
    elif quality_of_life >= 21: box11 =   0

    return box11


def get_box12(quality_of_life):

    # Box 12 Quality of Life-Politics 2-7
    if   quality_of_life <=  1: box12 = -10
    elif quality_of_life ==  2: box12 =  -8
    elif quality_of_life ==  3: box12 =  -6
    elif quality_of_life ==  4: box12 =  -3
    elif quality_of_life ==  5: box12 =  -2
    elif quality_of_life  < 10: box12 =  -1
    elif quality_of_life  < 12: box12 =   0
    elif quality_of_life  < 21: box12 =   1
    elif quality_of_life  < 24: box12 =   2
    elif quality_of_life  < 27: box12 =   3
    elif quality_of_life  < 29: box12 =   4
    elif quality_of_life >= 29: box12 =   5

    return box12


def get_boxD(quality_of_life):
    
    # Box D Quality of Life-Actionpoints 2-A
    if   quality_of_life <=  1: boxD = -6
    elif quality_of_life ==  2: boxD = -4
    elif quality_of_life ==  3: boxD = -2
    elif quality_of_life  <  6: boxD =  0
    elif quality_of_life  <  9: boxD =  1
    elif quality_of_life  < 18: boxD =  2
    elif quality_of_life  < 23: boxD =  3
    elif quality_of_life  < 27: boxD =  4
    elif quality_of_life >= 27: boxD =  5

    return boxD


def get_box13(population_growth):
    
    # Box 13 Population Growth-Population 3-6
    if   population_growth  <  3: box13 = -4
    elif population_growth  <  6: box13 = -3
    elif population_growth  < 10: box13 = -2
    elif population_growth  < 15: box13 = -1
    elif population_growth == 15: box13 =  0
    elif population_growth  < 21: box13 =  1
    elif population_growth  < 26: box13 =  2
    elif population_growth >= 26: box13 =  3

    return box13
            

def get_box14(population):
    
    # Box 14 Population-Quality of Life 6-2
    if   population  < 16: box14 =   0
    elif population  < 26: box14 =  -1
    elif population  < 31: box14 =  -2
    elif population  < 39: box14 =  -3
    elif population  < 43: box14 =  -4
    elif population  < 45: box14 =  -5
    elif population == 45: box14 =  -6
    elif population == 46: box14 =  -7
    elif population == 47: box14 =  -8
    elif population >= 48: box14 = -10

    return box14


def get_boxA(population):

    # Box A Population-Actionpoints 6
    if   population  < 10: boxA = 0
    elif population  < 17: boxA = 1
    elif population  < 22: boxA = 2
    elif population  < 26: boxA = 3
    elif population  < 30: boxA = 4
    elif population  < 34: boxA = 5
    elif population  < 38: boxA = 6
    elif population  < 42: boxA = 7
    elif population  < 46: boxA = 8
    elif population >= 46: boxA = 9

    return boxA


def get_boxW(population):
            
    # Box W Multiplier Population-Population 6-6
    if   population <=  1: boxW = 0
    elif population  < 15: boxW = 1
    elif population  < 36: boxW = 2
    elif population >= 36: boxW = 3

    return boxW
       

def get_boxB(politics):

    # Box B Politics-Actionpoints 7-A
    if   politics <= -10: boxB = -5
    elif politics ==  -9: boxB = -2
    elif politics  <  -2: boxB = -1
    elif politics  <   8: boxB =  0
    elif politics  <  22: boxB =  1
    elif politics  <  31: boxB =  2
    elif politics >=  31: boxB =  3
    
    return boxB
