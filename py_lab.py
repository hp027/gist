# -*- coding:utf-8 -*-

from sklearn.svm import SVC
__author__ = 'poo'


def iter():
    for i in range(10):
        yield i
        print "inline: ", i

class w_test():

    def __enter__(self):
        # raise Exception("a")
        # open("dsfasf")
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "type"
        return True


def a():
    count = 1
    def b():
        # nonlocal count
        print "call b"
    b()

if __name__ == '__main__':
    # try:
    with w_test() as w:
        open("sadfsad")
        print "a"
    # except:
        # pass

