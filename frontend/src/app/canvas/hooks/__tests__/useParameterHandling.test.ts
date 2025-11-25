import { renderHook, act } from '@testing-library/react';
import { useParameterHandling } from '../useParameterHandling';

describe('useParameterHandling', () => {
  describe('initialization', () => {
    it('should initialize with empty parameters by default', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      expect(result.current.parameters).toEqual({});
      expect(result.current.hasValidationErrors).toBe(false);
    });

    it('should initialize with provided parameters', () => {
      const initialParams = { kernelSize: 3, stride: 1 };
      const { result } = renderHook(() => 
        useParameterHandling({ initialParameters: initialParams })
      );
      
      expect(result.current.parameters).toEqual(initialParams);
    });
  });

  describe('handleParameterChange', () => {
    it('should update a single parameter', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleParameterChange('kernelSize', 5);
      });
      
      expect(result.current.parameters).toEqual({ kernelSize: 5 });
    });

    it('should update multiple parameters independently', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleParameterChange('kernelSize', 3);
      });
      
      act(() => {
        result.current.handleParameterChange('stride', 2);
      });
      
      expect(result.current.parameters).toEqual({ 
        kernelSize: 3, 
        stride: 2 
      });
    });

    it('should overwrite existing parameter value', () => {
      const { result } = renderHook(() => 
        useParameterHandling({ initialParameters: { kernelSize: 3 } })
      );
      
      act(() => {
        result.current.handleParameterChange('kernelSize', 5);
      });
      
      expect(result.current.parameters).toEqual({ kernelSize: 5 });
    });

    it('should handle different value types', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleParameterChange('intParam', 42);
        result.current.handleParameterChange('strParam', 'test');
        result.current.handleParameterChange('boolParam', true);
        result.current.handleParameterChange('arrParam', [1, 2, 3]);
      });
      
      expect(result.current.parameters).toEqual({
        intParam: 42,
        strParam: 'test',
        boolParam: true,
        arrParam: [1, 2, 3]
      });
    });
  });

  describe('handleValidationChange', () => {
    it('should update validation error state to true', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleValidationChange(true);
      });
      
      expect(result.current.hasValidationErrors).toBe(true);
    });

    it('should update validation error state to false', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleValidationChange(true);
      });
      
      act(() => {
        result.current.handleValidationChange(false);
      });
      
      expect(result.current.hasValidationErrors).toBe(false);
    });

    it('should toggle validation state multiple times', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleValidationChange(true);
      });
      expect(result.current.hasValidationErrors).toBe(true);
      
      act(() => {
        result.current.handleValidationChange(false);
      });
      expect(result.current.hasValidationErrors).toBe(false);
      
      act(() => {
        result.current.handleValidationChange(true);
      });
      expect(result.current.hasValidationErrors).toBe(true);
    });
  });

  describe('updateParameters', () => {
    it('should replace all parameters with new object', () => {
      const { result } = renderHook(() => 
        useParameterHandling({ initialParameters: { old: 'param' } })
      );
      
      act(() => {
        result.current.updateParameters({ 
          new: 'params', 
          another: 'value' 
        });
      });
      
      expect(result.current.parameters).toEqual({
        new: 'params',
        another: 'value'
      });
      expect(result.current.parameters).not.toHaveProperty('old');
    });

    it('should clear parameters when given empty object', () => {
      const { result } = renderHook(() => 
        useParameterHandling({ initialParameters: { param1: 1, param2: 2 } })
      );
      
      act(() => {
        result.current.updateParameters({});
      });
      
      expect(result.current.parameters).toEqual({});
    });
  });

  describe('integration scenarios', () => {
    it('should handle complete parameter lifecycle', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      // Set initial parameters
      act(() => {
        result.current.updateParameters({ kernelSize: 3, stride: 1 });
      });
      
      // Modify one parameter
      act(() => {
        result.current.handleParameterChange('kernelSize', 5);
      });
      
      expect(result.current.parameters).toEqual({ 
        kernelSize: 5, 
        stride: 1 
      });
      
      // Reset all parameters
      act(() => {
        result.current.updateParameters({ padding: 2 });
      });
      
      expect(result.current.parameters).toEqual({ padding: 2 });
    });

    it('should manage validation state independently of parameters', () => {
      const { result } = renderHook(() => useParameterHandling());
      
      act(() => {
        result.current.handleParameterChange('param', 'value');
        result.current.handleValidationChange(true);
      });
      
      expect(result.current.parameters).toEqual({ param: 'value' });
      expect(result.current.hasValidationErrors).toBe(true);
      
      act(() => {
        result.current.updateParameters({});
      });
      
      // Validation state should remain true after parameter update
      expect(result.current.parameters).toEqual({});
      expect(result.current.hasValidationErrors).toBe(true);
    });
  });

  describe('callback stability', () => {
    it('should maintain callback references across re-renders', () => {
      const { result, rerender } = renderHook(() => useParameterHandling());
      
      const firstHandleParameterChange = result.current.handleParameterChange;
      const firstHandleValidationChange = result.current.handleValidationChange;
      const firstUpdateParameters = result.current.updateParameters;
      
      rerender();
      
      expect(result.current.handleParameterChange).toBe(firstHandleParameterChange);
      expect(result.current.handleValidationChange).toBe(firstHandleValidationChange);
      expect(result.current.updateParameters).toBe(firstUpdateParameters);
    });
  });
});
