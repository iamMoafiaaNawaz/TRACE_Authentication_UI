import { API_BASE } from '../config/api';
const API_BASE_URL = API_BASE;

export async function submitAnalysis(file) {
  const formData = new FormData();
  formData.append("file", file);

  const token = localStorage.getItem("token");
  const headers = token ? { Authorization: `Bearer ${token}` } : {};

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers,
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error || "Server Error");
  }
  return data;
}



