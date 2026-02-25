import { useState } from 'react';
import { Listbox, ListboxButton, ListboxOptions, ListboxOption } from '@headlessui/react';
import { ChevronDown, Check, ArrowRight, Minus } from 'lucide-react';
import Card from '../components/Card';
import SliderControl from '../components/SliderControl';
import ChurnGauge from '../components/ChurnGauge';
import RiskFactors from '../components/RiskFactors';
import ErrorBanner from '../components/ErrorBanner';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  SLIDER_CONFIG,
  CONTRACT_OPTIONS,
  INTERNET_OPTIONS,
  PAYMENT_OPTIONS,
} from '../lib/constants';

function DropdownControl({ label, value, options, onChange }) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-slate-600">{label}</label>
      <Listbox value={value} onChange={onChange}>
        <div className="relative">
          <ListboxButton className="w-full flex items-center justify-between bg-white border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-700 shadow-sm hover:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-500">
            {value || 'Select...'}
            <ChevronDown size={14} className="text-slate-400" />
          </ListboxButton>
          <ListboxOptions className="absolute z-30 mt-1 w-full bg-white rounded-xl shadow-lg border border-slate-100 py-1">
            {options.map((opt) => (
              <ListboxOption
                key={opt}
                value={opt}
                className="flex items-center justify-between px-3 py-2 text-sm cursor-pointer data-[focus]:bg-brand-50 data-[selected]:bg-brand-100"
              >
                {opt}
                {opt === value && <Check size={13} className="text-brand-600" />}
              </ListboxOption>
            ))}
          </ListboxOptions>
        </div>
      </Listbox>
    </div>
  );
}

function SimulatePanel({ params, update, simResult, simLoading }) {
  const currentScore = simResult?.baseline_churn_probability;
  const projectedScore = simResult?.projected_churn_probability;
  const delta =
    projectedScore != null && currentScore != null
      ? projectedScore - currentScore
      : null;

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-slate-700">Quick Actions</h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Apply interventions to see projected churn impact
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => update({ ...params, Contract: 'Two year' })}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 transition-colors"
        >
          <ArrowRight size={14} />
          Upgrade to 2-year contract
        </button>

        <button
          onClick={() => {
            const next = Math.max(
              SLIDER_CONFIG.MonthlyCharges.min,
              params.MonthlyCharges - 15
            );
            update({ ...params, MonthlyCharges: next });
          }}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-700 transition-colors"
        >
          <Minus size={14} />
          Reduce charge by $15
        </button>

        <button
          onClick={() =>
            update({
              ...params,
              num_services: Math.min(SLIDER_CONFIG.num_services.max, params.num_services + 1),
            })
          }
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-500 text-white text-sm font-medium hover:bg-amber-600 transition-colors"
        >
          + Add TechSupport
        </button>
      </div>

      {/* Simulation result comparison */}
      {simLoading && <LoadingSpinner size="sm" />}

      {simResult && !simLoading && (
        <div className="grid grid-cols-3 gap-3 mt-2">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-center">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Baseline</p>
            <p className="text-xl font-bold text-slate-800">
              {currentScore != null ? `${(currentScore * 100).toFixed(1)}%` : '—'}
            </p>
          </div>
          <div className="flex items-center justify-center">
            <ArrowRight size={20} className="text-slate-400" />
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-center">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Projected</p>
            <p
              className="text-xl font-bold"
              style={{
                color:
                  delta == null
                    ? '#64748b'
                    : delta < 0
                    ? '#16a34a'
                    : '#dc2626',
              }}
            >
              {projectedScore != null ? `${(projectedScore * 100).toFixed(1)}%` : '—'}
            </p>
            {delta != null && (
              <p
                className="text-xs font-semibold mt-0.5"
                style={{ color: delta < 0 ? '#16a34a' : '#dc2626' }}
              >
                {delta < 0 ? '' : '+'}
                {(delta * 100).toFixed(1)}pp
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function WhatIfView({ hook }) {
  const { params, data, error, loading, pending, update, simResult, simLoading } = hook;

  const handleSlider = (key, value) => {
    update({ ...params, [key]: value });
  };

  const churnScore = data?.churn_probability ?? 0;
  const shapFactors = data?.shap_factors || data?.shap_values;

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 mb-1">Risk Simulator</h2>
        <p className="text-sm text-slate-500">
          Adjust subscriber features to explore predicted churn probability
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left: Controls */}
        <div className="space-y-4">
          <Card>
            <h3 className="text-base font-semibold text-slate-700 mb-4">Feature Inputs</h3>
            <div className="space-y-5">
              {Object.entries(SLIDER_CONFIG).map(([key, cfg]) => (
                <SliderControl
                  key={key}
                  label={cfg.label}
                  value={params[key]}
                  min={cfg.min}
                  max={cfg.max}
                  step={cfg.step}
                  onChange={(v) => handleSlider(key, v)}
                />
              ))}
            </div>
          </Card>

          <Card>
            <h3 className="text-base font-semibold text-slate-700 mb-4">Plan &amp; Payment</h3>
            <div className="space-y-4">
              <DropdownControl
                label="Contract Type"
                value={params.Contract}
                options={CONTRACT_OPTIONS}
                onChange={(v) => update({ ...params, Contract: v })}
              />
              <DropdownControl
                label="Internet Service"
                value={params.InternetService}
                options={INTERNET_OPTIONS}
                onChange={(v) => update({ ...params, InternetService: v })}
              />
              <DropdownControl
                label="Payment Method"
                value={params.PaymentMethod}
                options={PAYMENT_OPTIONS}
                onChange={(v) => update({ ...params, PaymentMethod: v })}
              />
            </div>
          </Card>
        </div>

        {/* Right: Gauge + Risk Factors */}
        <div className="space-y-4">
          <Card>
            <div className={`transition-opacity ${pending || loading ? 'opacity-50' : 'opacity-100'}`}>
              <div className="flex flex-col items-center py-2">
                <p className="text-xs text-slate-400 mb-4 uppercase tracking-wide font-medium">
                  Predicted Churn Probability
                </p>
                {loading ? (
                  <LoadingSpinner />
                ) : (
                  <ChurnGauge score={churnScore} size={200} />
                )}
              </div>
            </div>
          </Card>

          <ErrorBanner error={error} />

          {shapFactors && shapFactors.length > 0 && (
            <Card>
              <RiskFactors shapFactors={shapFactors} />
            </Card>
          )}
        </div>
      </div>

      {/* Simulate panel */}
      <Card>
        <SimulatePanel
          params={params}
          update={update}
          simResult={simResult}
          simLoading={simLoading}
        />
      </Card>
    </div>
  );
}
