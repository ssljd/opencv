import cv2

def mySift(a, b):
    sift = cv2.SIFT_create()
    kpa, desa = sift.detectAndCompute(a, None)
    kpb, desb = sift.detectAndCompute(b, None)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(desa, desb, k=2)
    good = [[m] for m, n in matches if m.distance < 0.6 * n.distance]
    result = cv2.drawMatchesKnn(a, kpa, b, kpb, good, None, flags=2)
    return result

if __name__ == '__main__':
    a = cv2.imread('../img/a.png')
    b = cv2.imread('../img/b.png')
    c = cv2.rotate(b, 0)
    m1 = mySift(a, b)
    m2 = mySift(a, c)
    m3 = mySift(b, c)
    cv2.imshow('a-b', m1)
    cv2.imshow('a-c', m2)
    cv2.imshow('b-c', m3)
    cv2.waitKey()
    cv2.destroyAllWindows()