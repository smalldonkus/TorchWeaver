import { useState, useEffect } from "react";

export interface VariableInfo {
  type: string;
  input_description: string;
}

export const useVariablesInfo = () => {
  const [variablesInfo, setVariablesInfo] = useState<VariableInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchVariablesInfo = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/variables-info');
        if (!response.ok) {
          throw new Error('Failed to fetch variables info');
        }
        const data = await response.json();
        setVariablesInfo(data.data || []);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching variables info:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    };

    fetchVariablesInfo();
  }, []);

  return {
    variablesInfo,
    loading,
    error
  };
};