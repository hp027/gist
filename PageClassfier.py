# -*- coding:utf-8 -*-
__author__ = 'poo'

import os
import re
import hashlib
import urllib
import threading
import pickle
import sys
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")


def md5(org):
    m5 = hashlib.md5()
    m5.update(org)
    return m5.hexdigest()


# 改非加密模式
def repath(url):
    return url.replace("https://", "http://")


# 网页抓取线程
class WebpageFetchThread(threading.Thread):
    url = []
    dist = []

    def __init__(self, url, dist):
        threading.Thread.__init__(self)
        self.url = url
        self.dist = dist

    def run(self):
        try:
            resp = urllib.urlopen(self.url).read()
            try:
                resp = resp.decode("gbk")
            except:
                pass
            f = file(self.dist, "w")
            f.write(resp)
            f.close()
        except IOError:
            print "IOError: (URL: %s)" % self.url


# 读取书签格式数据
def load_pages_from_bookmark(path):
    f = file(path)
    pages = []
    for l in f.xreadlines():
        '''
            Format: "name": "openlayers 加载 arcgis rest服务_啥都有","url": "http://lyj289.diandian.com/post/2012-04-19/19071138"
        '''
        # print l[9:-2]
        ss = l[9:-2].split("\",\"url\": \"")
        if len(ss) != 2: continue
        o = (ss[1], ss[0])
        pages.append(o)
    return pages


# 读取csv格式数据
def load_pages_from_csv(path):
    f = file(path)
    isTitle = True
    pages = []
    for l in f.xreadlines():
        if isTitle:
            isTitle = False
            continue
        ls = l.split(", ")
        if len(ls) != 8: continue
        o = (ls[1].replace("\"", ""), ls[2].replace("\"", ""))
        pages.append(o)
    return pages


# 生成训练index
def generate_page_index(pages, pkg, dist_path="pages"):
    # pages = loadPagesFromBookmark("bookmarks")
    print "Items count: %s" % len(pages)
    # tp = pages[50]
    pobjs = []
    keys = {}
    for url, title in pages:
        m5 = md5(url)
        if os.path.exists("%s/%s" % (dist_path, m5)) and m5 not in keys:
            keys[m5] = "";
            pobjs.append({'url': url, 'title': title, "md5": m5})
    pickle.dump(pobjs, file(pkg, "wb"))
    print "Filter count: %s" % len(pobjs)


# 读取训练index
def load_page_index(pkg):
    return pickle.load(file(pkg, "rb"))
    # pass


# 特征抽取工具
class FeaturesToolkit():
    url, title, md5, html = "", "", "", ""
    dist_path = ""
    features = {}

    def __init__(self, page, dist_path="pages"):
        self.url = page['url']
        self.title = page['title']
        self.md5 = page['md5']
        self.dist_path = dist_path
        pass

    def get_features(self, is_lite=False):

        return self._get_lite_features() if is_lite else self.features

    def _get_lite_features(self):
        f = self.features
        # print f
        return f['url'] \
               + [f['html']['length']] + f['html']['tag_features'] \
               + [f['title']['length']] + [int(f['title']['is_with_sub'])] + [int(f['title']['is_with_col'])] + [
                   f['title']['count_of_punc']] + [f['title']['count_of_brace']]

    def analysis(self):
        # self._anal_url_features(self.url)
        #
        # self._anal_title_features(self.title)
        try:
            self.html = file("%s/%s" % (self.dist_path, self.md5)).read()
        except:
            print self.url
        # print self._anal_html_features(self.html)
        # print self.title
        self.features = {
            "url": self._anal_url_features(self.url),
            "title": self._anal_title_features(self.title),
            "html": self._anal_html_features(self.html)
        }
        return self
        # pass

    def _anal_url_features(self, url):
        # print url
        url = url.replace("https://", "").replace("http://", "")
        loc, search = url, ""

        is_www_start = "www." in url
        is_with_search = "?" in url
        if is_with_search:
            loc, search = url.split("?")[:2]
        is_search_split = "&" in url
        loc_isnum_segs = loc.replace("-", "#").replace("_", "#").replace("/", "#").split("#")
        seg_of_num = 0
        for s in loc_isnum_segs:
            try:
                int(s)
            except:
                continue
            seg_of_num += 1

        if loc.endswith("/"):
            loc = loc[:-1]
        is_end_with_html = loc.endswith(".html") or loc.endswith(".htm")
        seg_of_loc = len(loc.split("/"))

        seg_of_search = len(search.split("&")) if is_with_search else 0

        res = {"is_www_start": is_www_start,
               "is_with_search": is_with_search,
               "is_search_split": is_search_split,
               "is_end_with_html": is_end_with_html,
               "seg_of_num": seg_of_num,
               "seg_of_loc": seg_of_loc,
               "seg_of_search": seg_of_search}
        lite_res = map(lambda i: int(i) if type(i) == bool else i,
                       [is_www_start, is_with_search, is_search_split, is_end_with_html, seg_of_num, seg_of_loc,
                        seg_of_search])

        # print res
        return lite_res

    def _anal_title_features(self, title):
        # 解决中文匹配问题
        BRACE_REGX = u"[（|）|【|】|[|]|<|>|【|】]|(－{2})|[(|)]|(（）)|(【】)|({})|(《》)|(-{2})|(/.{3})|(/(/))|(/[/])|({})"
        PUNC_REGX = u"[_,+-.?:;'\"!`，。？：；‘’！“”—……、]"
        # print title
        return {
            "length": len(self.title),
            "is_with_sub": "-" in self.title or "_" in self.title,
            "is_with_col": ":" in self.title,
            "count_of_brace": len(re.findall(BRACE_REGX, self.title.decode("utf-8"))),
            "count_of_punc": len(re.findall(PUNC_REGX, self.title.decode("utf-8"))),
        }
        pass

    def _anal_html_features(self, html):
        global TAGS
        if len(TAGS) == 0:
            TAGS = load_tags()
        return {
            "length": len(html),
            'tag_features': self._anal_html_tag_features(TAGS, html)
        }

    def _anal_html_tag_features(self, tags, html):
        return [len(re.findall("<%s[ |>]" % tag, html)) for tag in tags]


# 多线程获取网页内容
def fetch_pages_content(pages, dist_path="pages"):
    print "Fetching processing..."
    for url, title in pages:
        output = "%s/%s" % (dist_path, md5(url))
        if not os.path.exists(output):
            # print url
            WebpageFetchThread(repath(url), output).start()
            pass
            # print "Fetching finished!"


TAGS = []


# 读取标签
def load_tags(path="tags"):
    return [t[:-1] for t in open(path, "r").readlines()]


from sklearn import cluster, datasets
import sklearn

import readability


# 使用readability测试
def readability_test(idxs, dist_path="pages"):
    lite_pages = []
    fat_pages = []
    for idx in idxs:
        c = file("%s/%s" % (dist_path, idx['md5'])).read()
        l = len(readability.Document(idx['url']).summary())
        if l < 200:
            lite_pages.append((l, idx['url']))
        elif l > 400:
            fat_pages.append((l, idx['url']))
            # print idx['url']
    for l in lite_pages:
        print l
    print "________________________________________________"
    for f in fat_pages:
        print f


# 测试分类器-kmeans
def clf_test(idx, dist_path="pages"):
    train_data = []

    for pi in idx:
        features = FeaturesToolkit(pi, dist_path=dist_path).analysis().get_features(is_lite=False)['url']
        train_data.append(features)

    clsK = 30
    kmClf = cluster.KMeans(clsK)
    kmClf.fit(train_data)
    labels = kmClf.labels_
    # print kmClf.labels_[::20]
    clt = {}
    # keep site alone
    keys = {}
    for i, l in enumerate(labels):
        l = str(l)
        if l not in clt:
            clt[l] = []
            keys[l] = {}
        key = ""
        try:
            key = idx[i]['url'].split("//")[1].split("/")[0]
        except:
            pass
        if key not in keys[l]:
            keys[l][key] = ""
            clt[l].append(idx[i]['url'])

    for i in range(clsK):
        print "CLS: %d \t COUNT: %d" % (i, len(clt[str(i)]))
        print "\n".join(clt[str(i)][:20])


# 读取标记数据
def load_clf_label(features=None):
    ls = open("pages_classfied").readlines()
    clf_url, clf_labels, clf_features, label_descs, label = [], [], [], {}, 0
    for l in ls:
        l = l[:-1]
        if l.startswith("###") and l.endswith("###"):
            label += 1
            label_descs[str(label)] = l[3: -3]
            continue
        # try:
        if not len(l.replace(" ", "").replace("\t", "")) == 0:
            clf_url.append(l)
            clf_labels.append(label)
            if features:
                clf_features.append(features[l])
                # except:
                #     print "ERROR: %s" % l
    return (clf_url, clf_features, clf_labels, label_descs)


def load_train_data():
    urls, tr_features, rel_labels, descs = load_clf_label()
    tr_features, labels = [], []
    tfs = None
    for i, url in enumerate(urls):
        try:
            # 若分析features异常，跳过
            fs = FeaturesToolkit(fetch_index(url, "cache", False), "cache").analysis().get_features(is_lite=True)
            # if url == "http://www.jd.com/":
            #     title = fetch_index(url, "cache", False)['title']
            #     tfs = fs
            #     print len(title), title, fs
            tr_features.append(fs)
            labels.append(rel_labels[i])
        except:
            # print "Features fetched ERROR: URL: %s MD5: %s" % (url, md5(url))
            pass
    return tr_features, labels


def train_SVC_clf_test():
    features, labels = load_train_data()
    clf = train_SVC_clf(features, labels)
    # print clf.predict(tfs)
    rate = sum(np.array(clf.predict(features)) == np.array(labels)) * 1.0 / len(labels)
    return clf, rate


import urllib2
import StringIO
import gzip


# 带header请求数据
def fetch_content(url):
    # gzip压缩获取
    UA = "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"
    headers = {'Accept-encoding': 'gzip, deflate', 'User-Agent': UA}
    headers_without_gzip = {'Accept-encoding': '', 'User-Agent': UA}
    req = urllib2.Request(url, headers=headers)
    bin = urllib2.urlopen(req).read()
    try:
        content = gzip.GzipFile(fileobj=StringIO.StringIO(bin)).read()
    except IOError as e:
        content = bin
        if "CRC check" in e.message:
            content = urllib2.urlopen(urllib2.Request(url, headers=headers_without_gzip)).read()
    return content


# 训练SVR分类器
def train_SVC_clf(features, labels):
    svr = sklearn.svm.SVC(kernel='rbf', C=1.5, gamma=1e-08)
    return svr.fit(features, labels)


# 网页解析异常
class HTMLParseException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# 根据url获取index
def fetch_index(url, dist_path="pages", is_rewrite=True):
    m5 = md5(url)
    if not is_rewrite and os.path.exists("%s/%s" % (dist_path, m5)):
        html = open("%s/%s" % (dist_path, m5)).read()
    else:
        html = fetch_content(url)
        try:
            html = html.decode("gbk")
        except:
            pass
        # TODO： 文件读写前后获取数据不一致，强制重写var html， 怀疑linux编码
        open("%s/%s" % (dist_path, m5), "w").write(html)
        html = open("%s/%s" % (dist_path, m5), "r").read()
    # print url
    # print m5
    try:
        title = html[html.index("<title>") + 7: html.index("</title>")]
    except:
        raise HTMLParseException("REGX_TITLE_EXCEPTION")
        # title = ""

    return {'url': url, 'title': title, "md5": m5}


fts, lts = [], []


def scorer(estimator, X, y):
    # return estimator.score(X, y)
    return estimator.score(fts, lts)


from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV as GSCV


def best_clf_test():
    fs, ls = load_train_data()
    global fts, lts
    frs, fts, lrs, lts = cross_validation.train_test_split(fs, ls, test_size=0.4, random_state=0)
    print len(frs), len(fts)
    param_grid = {'C': [1, 1.2, 1.25, 1.3, 1.45, 1.4, 1.5, 1.6, 1.7, 2, 5, 8, 10],
                  'gamma': [0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10], 'kernel': ['rbf']}
    clf = GSCV(sklearn.svm.SVC(), param_grid, cv=5, scoring=scorer)
    clf.fit(frs, lrs)
    print clf.best_score_
    print clf.best_params_
    b_clf = clf.best_estimator_
    print b_clf.predict(fs)


# ecc9510e960297cf91c124d3b635dfd1
# 获取测试数据
def test():
    global TAGS
    TAGS = load_tags()
    best_clf_test()
    # print TAGS
    # pages = load_pages_from_csv("urls_u330p.csv")
    # fetch_pages_content(pages, "pages_u330p")

    # generate_page_index(pages, "pages_u330p.idx", "pages_u330p")


    # idx = load_page_index("pages_u330p.idx")



    # pkg = "pages"
    # feature_objs = {}
    # idx = load_page_index("%s.idx" % pkg)
    # for idx in idx:
    #     feature = FeaturesToolkit(idx, pkg).analysis().get_features(is_lite = True)
    #     feature_objs[idx['url']] = feature
    #
    # pkg = "pages_u330p"
    # feature_objs = {}
    # idx = load_page_index("%s.idx" % pkg)
    # for idx in idx:
    #     feature = FeaturesToolkit(idx, pkg).analysis().get_features(is_lite = True)
    #     feature_objs[idx['url']] = feature

    # idx =load_page_index("pages.idx")
    # clf_test(idx, "pages")

    # fss = pickle.load(open("features.pkg"))
    # url = "http://iec.qdu.edu.cn/content-16-556-1.html"
    # print md5(url)
    # print fss[url]
    # fs = FeaturesToolkit(fetch_index(url, "cache"), "cache").analysis().get_features(is_lite = True)
    # print fs
    # clf = pickle.load(open("page.clf"))
    #
    # print clf.predict(fs)

    # clf, rate = train_SVC_clf_test()
    # print "RATE: %s" % str(rate)
    # pickle.dump(clf, open("page.clf", "wb"))




    # print fs

    # print "loading features pkg"
    # features = pickle.load(open("features.pkg", "rb"))
    # print "trian data predict"
    # urls, tr_features, labels = load_clf_label(features)
    # print "test data predict"
    #
    # clf = train_SVC_clf(tr_features, labels)
    # print clf.predict(tr_features)
    # pickle.dump(clf, open("page.clf", "wb"))
    # test_url, test_features = features.keys(), features.values()
    # ls = clf.predict(test_features)
    # ls_cluster = {}
    # for i, l in enumerate(ls):
    #     if l not in ls_cluster:
    #         ls_cluster[l] = []
    #     ls_cluster[l].append(test_url[i])
    # for l in ls_cluster.keys():
    #     print "CLS: %s\tCOUNT: %s" % (l, len(ls_cluster[l]))
    #     for s in ls_cluster[l][::15][:20]:
    #         print s








    # 111 features
    # print feature
    # print len(feature_objs.keys())
    # pickle.dump(feature_objs, open("features.pkg", "wb"))
    # print len(features)
    # print fs
    # print fs
    # print (fs["title"]["count_of_brace"],fs["title"]["count_of_punc"])
    # clf_test(idx, "pages_u330p")
    # readability_test(idx, pkg)


if __name__ == "__main__":
    test()
