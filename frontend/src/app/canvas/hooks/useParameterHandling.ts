import { useState, useCallback } from "react";

interface UseParameterHandlingProps {
  initialParameters?: Record<string, any>;
}

export const useParameterHandling = ({ initialParameters = {} }: UseParameterHandlingProps = {}) => {
  const [parameters, setParameters] = useState(initialParameters);
  const [hasValidationErrors, setHasValidationErrors] = useState(false);

  const handleParameterChange = useCallback((parameterKey: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [parameterKey]: value
    }));
  }, []);

  const handleValidationChange = useCallback((hasErrors: boolean) => {
    setHasValidationErrors(hasErrors);
  }, []);

  const updateParameters = useCallback((newParameters: Record<string, any>) => {
    setParameters(newParameters);
  }, []);

  return {
    parameters,
    hasValidationErrors,
    handleParameterChange,
    handleValidationChange,
    updateParameters
  };
};