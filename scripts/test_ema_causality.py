#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

import pa_engine as pe
import backtest as bt

def test_ema_causality():
    # Verify that ewm(span=200, adjust=False) is perfectly causal and identical across sub-slices
    np.random.seed(42)
    s = pd.Series(100.0 + np.cumsum(np.random.randn(700)))
    
    ema_full = s.ewm(span=200, adjust=False).mean()
    
    for T in [300, 450, 600]:
        sub_s = s.iloc[:T+1]
        ema_sub = sub_s.ewm(span=200, adjust=False).mean()
        
        diff = np.max(np.abs(ema_full.iloc[:T+1].values - ema_sub.values))
        print(f"Max diff at T={T}: {diff:.2e}")
        assert diff < 1e-12, f"EMA differs at T={T}!"

    print("EMA CAUSALITY 100% VERIFIED!")

if __name__ == "__main__":
    test_ema_causality()
