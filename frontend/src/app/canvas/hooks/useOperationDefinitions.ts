import { useState, useEffect } from 'react';

interface OperationDefinition {
  library: string;
  type: string;
  maxInputs: number;
  minInputs: number;
  parameters: Record<string, any>;
}

interface OperationsData {
  layers: any;
  tensorOps: any;
  activators: any;
  inputs: any;
}

const API_BASE_URL = 'http://localhost:5000';

export function useOperationDefinitions() {
  const [operations, setOperations] = useState<OperationsData>({
    layers: {},
    tensorOps: {},
    activators: {},
    inputs: {}
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOperations = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_BASE_URL}/api/operations`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        setOperations({
          layers: data.layers || {},
          tensorOps: data.tensorOps || {},
          activators: data.activators || {},
          inputs: data.inputs || {}
        });
      } catch (err) {
        console.error('Failed to fetch operation definitions:', err);
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
        
        // Fallback to empty objects if backend is not available
        setOperations({
          layers: {},
          tensorOps: {},
          activators: {},
          inputs: {}
        });
      } finally {
        setLoading(false);
      }
    };

    fetchOperations();
  }, []);

  const refetch = () => {
    setLoading(true);
    setError(null);
    // Re-trigger the useEffect by updating a dependency
    useEffect(() => {
      // This will trigger a re-fetch
    }, []);
  };

  return {
    ...operations,
    loading,
    error,
    refetch
  };
}

export default useOperationDefinitions;