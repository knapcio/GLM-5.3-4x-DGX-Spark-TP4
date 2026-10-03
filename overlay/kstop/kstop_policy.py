# SPDX-License-Identifier: Apache-2.0
"""One confidence policy, independent of prompt class and generated phase."""
import math


def lengths(confidence, tau=.74):
    if not math.isfinite(tau) or not .6 <= tau <= .85:
        raise ValueError('tau must be finite in [.6,.85]')
    out=[]
    for row in confidence:
        if len(row)<2 or any(not math.isfinite(p) or not 0<=p<=1 for p in row[:2]):
            raise ValueError('invalid native top1 probability')
        out.append(1 if row[0]<tau else 2 if row[0]*row[1]<tau else 3)
    return out


def stop(lengths_, cumulative, probability, step, tau):
    """After pass step: keep its token, then choose whether to execute next."""
    if step not in (1,2) or not len(lengths_)==len(cumulative)==len(probability):
        raise ValueError('boundary shape/step')
    out=list(lengths_); cum=list(cumulative)
    for i,p in enumerate(probability):
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('invalid native top1 probability')
        if out[i]>step:
            cum[i]*=p
            if cum[i]<tau:out[i]=step
    return out,cum


def sources(widths, kept):
    """Preserve rows 0..K inclusive; later rows reuse same-request root."""
    if len(widths)!=len(kept):raise ValueError('shape')
    src=[];dead=[];start=0
    for w,k in zip(widths,kept):
        if not 1<=k<w:raise ValueError('live bonus row required')
        for j in range(w):src.append(start+j if j<=k else start);dead.append(j>k)
        start+=w
    return src,dead
