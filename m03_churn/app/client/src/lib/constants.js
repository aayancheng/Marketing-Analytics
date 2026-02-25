export const RISK_TIERS = ["High Risk", "Medium-High Risk", "Medium-Low Risk", "Low Risk"];

export const RISK_COLORS = {
  "High Risk": "#ef4444",        // red-500
  "Medium-High Risk": "#f97316", // orange-500
  "Medium-Low Risk": "#eab308",  // yellow-500
  "Low Risk": "#22c55e",         // green-500
};

export const SLIDER_CONFIG = {
  tenure: { min: 0, max: 72, step: 1, label: "Tenure (months)", default: 32 },
  MonthlyCharges: { min: 18, max: 120, step: 0.5, label: "Monthly Charges ($)", default: 65 },
  num_services: { min: 0, max: 8, step: 1, label: "Active Services", default: 4 },
  TotalCharges: { min: 0, max: 8000, step: 10, label: "Total Charges ($)", default: 2000 },
};

export const CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"];
export const INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"];
export const PAYMENT_OPTIONS = [
  "Electronic check",
  "Mailed check",
  "Bank transfer (automatic)",
  "Credit card (automatic)",
];

export const DEFAULT_WHATIF = {
  tenure: 32,
  MonthlyCharges: 65,
  num_services: 4,
  TotalCharges: 2000,
  Contract: "Month-to-month",
  InternetService: "Fiber optic",
  PaymentMethod: "Electronic check",
};

export function interpolateRiskColor(score) {
  // 0 = green, 0.5 = yellow, 1 = red
  if (score < 0.5) {
    const t = score * 2;
    const r = Math.round(34 + (234 - 34) * t);
    const g = Math.round(197 + (179 - 197) * t);
    const b = Math.round(94 + (8 - 94) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (score - 0.5) * 2;
    const r = Math.round(234 + (239 - 234) * t);
    const g = Math.round(179 + (68 - 179) * t);
    const b = Math.round(8 + (68 - 8) * t);
    return `rgb(${r},${g},${b})`;
  }
}

export function getRiskTier(score) {
  if (score >= 0.75) return "High Risk";
  if (score >= 0.5) return "Medium-High Risk";
  if (score >= 0.25) return "Medium-Low Risk";
  return "Low Risk";
}

export function formatError(error) {
  const msgs = {
    not_found: "Customer not found",
    invalid_id: "Invalid customer ID format",
    insufficient_data: "Insufficient customer data",
  };
  return msgs[error?.error] || error?.message || msgs[error] || "An unexpected error occurred";
}
