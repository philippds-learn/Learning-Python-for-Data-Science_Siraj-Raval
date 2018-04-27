#https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU

from sklearn import tree

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
#X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
#     [190, 90, 47], [175, 64, 39],
#     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
#     'female', 'male', 'male']

# [area occupied, 4 x 4 matrix]
X1 = [
#AI
[10,10,0,0,
10,10,0,0,
0,0,0,0],

#PATH
[0,0,10,0,
0,0,10,0,
10,0,0,0],

#CODE
[0,0,0,0,
0,0,0,0,
0,10,10,0],

#VR
[0,0,0,10,
0,0,0,10,
0,0,0,10]]

Y1 = [
'AI',
'PATH',
'CODE',
'VR']

print(len(X1))
print(len(Y1))

# CHALLENGE - ...and train them on our data
clf = clf.fit(X1, Y1)
#prediction = clf.predict([[160, 62, 39]])
prediction = clf.predict([[ 0,0,0,9,
                            0,0,0,9,
                            0,9,9,9]])
# CHALLENGE compare their reusults and print the best one!

print(prediction)
