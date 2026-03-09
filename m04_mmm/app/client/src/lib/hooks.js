import { useState, useEffect, useRef, useCallback } from 'react';
import {
  fetchDecomposition,
  fetchRoas,
  fetchResponseCurves,
  fetchAdstock,
  fetchOptimalAllocation,
  simulateBudget,
} from './api';

export function useDecomposition() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDecomposition()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { data, loading, error };
}

export function useChannelPerformance() {
  const [roas, setRoas] = useState(null);
  const [curves, setCurves] = useState(null);
  const [adstockData, setAdstockData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([fetchRoas(), fetchResponseCurves(), fetchAdstock()])
      .then(([r, c, a]) => {
        setRoas(r);
        setCurves(c);
        setAdstockData(a);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { roas, curves, adstockData, loading, error };
}

export function useSimulator() {
  const [spends, setSpends] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState(null);
  const [budgetLock, setBudgetLock] = useState(false);
  const timerRef = useRef(null);

  // Load initial spends from optimal-allocation current values
  useEffect(() => {
    fetchOptimalAllocation()
      .then((alloc) => {
        const initial = {
          tv_spend: alloc.current.tv || 0,
          ooh_spend: alloc.current.ooh || 0,
          print_spend: alloc.current.print || 0,
          facebook_spend: alloc.current.facebook || 0,
          search_spend: alloc.current.search || 0,
        };
        setSpends(initial);
      })
      .catch((e) => setError(e.message))
      .finally(() => setInitialLoading(false));
  }, []);

  // Debounced simulate call
  useEffect(() => {
    if (!spends) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setLoading(true);
      simulateBudget(spends)
        .then(setResult)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(timerRef.current);
  }, [spends]);

  const totalBudget = spends
    ? Object.values(spends).reduce((s, v) => s + v, 0)
    : 0;

  const updateSpend = useCallback(
    (key, value) => {
      setSpends((prev) => {
        if (!prev) return prev;
        if (!budgetLock) {
          return { ...prev, [key]: value };
        }
        // Budget lock: proportionally scale other channels
        const oldVal = prev[key];
        const delta = value - oldVal;
        const otherKeys = Object.keys(prev).filter((k) => k !== key);
        const otherTotal = otherKeys.reduce((s, k) => s + prev[k], 0);
        if (otherTotal === 0) return { ...prev, [key]: value };
        const next = { ...prev, [key]: value };
        otherKeys.forEach((k) => {
          const ratio = prev[k] / otherTotal;
          next[k] = Math.max(0, Math.round(prev[k] - delta * ratio));
        });
        return next;
      });
    },
    [budgetLock]
  );

  return {
    spends,
    result,
    loading,
    initialLoading,
    error,
    budgetLock,
    setBudgetLock,
    updateSpend,
    totalBudget,
  };
}

export function useOptimalAllocation() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchOptimalAllocation()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { data, loading, error };
}
