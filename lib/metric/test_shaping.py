from RewardShaping import *

func = RewardShaping("bin", 5)
print func.param
print "Binning: "
for i in np.arange(0, 1, 0.05):
    print i, " ---> ", func(i)
print

print "Noise: "
func = RewardShaping("noise", 0.1)
print func.param
for i in xrange(10):
    r = 0.3
    print r, " ---> ", func(r)
print


def test_curve(func):
    print func.param
    for i in np.arange(0, 1, 0.1):
        print i, " ---> ", func(i), "Diff = ", i - func(i)
    print

print "Curving: "
func = RewardShaping("curve", 1.1)
test_curve(func)
func = RewardShaping("curve", 0.9)
test_curve(func)
func = RewardShaping("curve", 0.8)
test_curve(func)

func = RewardShaping("curve", 1.2)
test_curve(func)

func = RewardShaping("curve", 0.5)
test_curve(func)

func = RewardShaping("curve", 1.5)
test_curve(func)

