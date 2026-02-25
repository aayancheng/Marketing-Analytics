import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchCustomer,
  fetchAllCustomers,
  fetchPrediction,
  fetchSimulation,
  fetchCustomers,
  fetchSegments,
} from './api';
import { DEFAULT_WHATIF, RISK_TIERS } from './constants';

export function useCustomer() {
  const [customerId, setCustomerId] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [allCustomers, setAllCustomers] = useState([]);
  const [customersLoading, setCustomersLoading] = useState(false);

  const lookup = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchCustomer(id);
      setData(result);
    } catch (e) {
      setData(null);
      setError(e.response?.data?.detail || { error: 'unknown', message: 'Request failed' });
    } finally {
      setLoading(false);
    }
  }, []);

  const loadAll = useCallback(async () => {
    setCustomersLoading(true);
    try {
      const customers = await fetchAllCustomers();
      setAllCustomers(customers);
    } finally {
      setCustomersLoading(false);
    }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  return { customerId, setCustomerId, data, error, loading, lookup, allCustomers, customersLoading };
}

export function usePredict() {
  const [params, setParams] = useState(DEFAULT_WHATIF);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState(false);
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const timerRef = useRef(null);

  const update = useCallback((next) => {
    setParams(next);
    setPending(true);
    setError(null);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setPending(false);
      setLoading(true);
      try {
        const result = await fetchPrediction(next);
        setData(result);
      } catch {
        setError({ message: 'Prediction failed. Please try again.' });
      } finally {
        setLoading(false);
      }
    }, 300);
  }, []);

  const simulate = useCallback(async (customerId, changes) => {
    setSimLoading(true);
    try {
      const result = await fetchSimulation(customerId, changes);
      setSimResult(result);
    } catch {
      setSimResult(null);
    } finally {
      setSimLoading(false);
    }
  }, []);

  // Initial prediction on mount
  useEffect(() => { update(DEFAULT_WHATIF); }, [update]);

  return { params, data, error, loading, pending, update, simResult, simLoading, simulate };
}

export function useSegments() {
  const [segment, setSegment] = useState(RISK_TIERS[0]);
  const [customers, setCustomers] = useState([]);
  const [summary, setSummary] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadCustomers = useCallback(async (seg) => {
    setLoading(true);
    try {
      const resp = await fetchCustomers(1, 100, seg);
      setCustomers(resp.items || []);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSummary = useCallback(async () => {
    try {
      const resp = await fetchSegments();
      setSummary(resp.segments || []);
    } catch {
      // non-critical
    }
  }, []);

  useEffect(() => { loadCustomers(segment); }, [segment, loadCustomers]);
  useEffect(() => { loadSummary(); }, [loadSummary]);

  const changeSegment = useCallback((seg) => setSegment(seg), []);

  return { segment, changeSegment, customers, summary, loading };
}
