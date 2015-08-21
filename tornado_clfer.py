# -*- coding:utf-8 -*-
__author__ = 'poo'

import pickle
import tornado.ioloop
import tornado.web
import PageClassfier
from PageClassfier import load_clf_label
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

clf = None
label_descs = {}


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class PredictHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        global clf
        refer = self.request.headers['Referer']
        # print refer, PageClassfier.md5(refer)
        try:
            idx = PageClassfier.fetch_index(refer, "cache")
            fs = PageClassfier.FeaturesToolkit(idx, "cache").analysis().get_features(
            is_lite=True)
            title = PageClassfier.fetch_index(refer, "cache", False)['title']

            cls = clf.predict(fs)[0]
            # print cls, len(title), title, fs
            print "URL: %s\nMD5:%s\tCLS: %s" % (refer, PageClassfier.md5(refer), cls)
            resp = {"result": True, "cls": cls, "desc": label_descs}
        except PageClassfier.HTMLParseException as e:
            resp = {"result": False, "msg":e.value}
        # print fs

        self.write(json.dumps(resp, ensure_ascii=False))

import re
def modify_label(refer, label):
    i = open("pages_classfied", "r")
    res = i.read()
    i.close()
    # 已存在先删除
    try:
        res.index("\n%s\n" % refer)
        res = res.replace("%s\n" % refer, "")
    except:
        pass
    # 找标签位置
    label_tag = re.findall("###[\w\W]*?###\n", res)[int(label) - 1]
    res = res.replace(label_tag, "%s%s\n" % (label_tag, refer))
    o = open("pages_classfied", "w")
    o.write(res)
    o.close()
    return True


class ModifyHandler(tornado.web.RequestHandler):
    def get(self):
        refer = self.request.headers['Referer']
        label = self.get_argument("label")
        self.write(json.dumps({"result":modify_label(refer, label)}))

class RetrainHandler(tornado.web.RequestHandler):
    def get(self):
        global clf
        clf, rate = PageClassfier.train_SVC_clf_test()
        pickle.dump(clf, open("page.clf", "wb"))

        self.write(json.dumps({"result":True, "msg": "Rate: %.4f" % rate}))

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/predict", PredictHandler),
    (r"/modify", ModifyHandler),
    (r"/retrain", RetrainHandler)
])

def reload_clf():
    global clf
    clf = pickle.load(open("page.clf", "rb"))

if __name__ == "__main__":
    print "predict server running..."
    global label_descs
    reload_clf()
    label_descs = load_clf_label()[-1]
    application.listen(9999)
    tornado.ioloop.IOLoop.instance().start()
