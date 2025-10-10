# sync_estimate.py
import cv2 as cv, numpy as np

LEFT="../video/new_output_left.MP4"; RIGHT="../video/new_output_right.MP4"
SAMPLE=1200  
STEP=10     
MAX_LAG=10  

def frame_diffs(path, sample, step):
    cap=cv.VideoCapture(path); n=int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    idxs=list(range(0, min(n, sample*step), step))
    vals=[]
    prev=None
    for i in idxs:
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ok, f=cap.read()
        if not ok: break
        g=cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        g=cv.GaussianBlur(g,(5,5),0)
        if prev is not None:
            vals.append(float(np.mean(cv.absdiff(g, prev))))
        prev=g
    cap.release()
    return np.array(vals)

a=frame_diffs(LEFT, SAMPLE, STEP)
b=frame_diffs(RIGHT,SAMPLE, STEP)

lags=range(-MAX_LAG, MAX_LAG+1)
best_lag=0; best_corr=-1e9
for lag in lags:
    if lag>=0:
        x=a[lag:len(b)+lag if len(a)>=len(b)+lag else len(a)]
        y=b[:len(x)]
    else:
        x=a[:len(a)+lag]
        y=b[-lag:len(b)]
    if len(x)<10: continue
    c=np.corrcoef(x,y)[0,1]
    if c>best_corr: best_corr=c; best_lag=lag
print("Estimated frame lag (LEFT relative to RIGHT):", best_lag, "frames; corr=", best_corr)
