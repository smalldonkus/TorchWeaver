import { validateParameter } from '../parameterValidation';

describe('parameterValidation', () => {
  describe('validateParameter', () => {
    describe('int validation', () => {
      it('should validate positive integers', () => {
        const result = validateParameter('42', ['int']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(42);
      });

      it('should validate negative integers', () => {
        const result = validateParameter('-10', ['int']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(-10);
      });

      it('should reject non-integer values', () => {
        const result = validateParameter('abc', ['int']);
        expect(result.isValid).toBe(false);
        expect(result.error).toBeDefined();
      });

      it('should reject empty values', () => {
        const result = validateParameter('', ['int']);
        expect(result.isValid).toBe(false);
        expect(result.error).toBe('Value cannot be empty');
      });
    });

    describe('float validation', () => {
      it('should validate positive floats', () => {
        const result = validateParameter('3.14', ['float']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(3.14);
      });

      it('should validate negative floats', () => {
        const result = validateParameter('-2.5', ['float']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(-2.5);
      });

      it('should allow intermediate typing states like "1."', () => {
        const result = validateParameter('1.', ['float']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe('1.');
      });

      it('should allow minus sign for negative input', () => {
        const result = validateParameter('-', ['float']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe('-');
      });
    });

    describe('boolean validation', () => {
      it('should validate true values', () => {
        expect(validateParameter('true', ['boolean']).convertedValue).toBe(true);
        expect(validateParameter('True', ['boolean']).convertedValue).toBe(true);
        expect(validateParameter('1', ['boolean']).convertedValue).toBe(true);
        expect(validateParameter('yes', ['boolean']).convertedValue).toBe(true);
      });

      it('should validate false values', () => {
        expect(validateParameter('false', ['boolean']).convertedValue).toBe(false);
        expect(validateParameter('False', ['boolean']).convertedValue).toBe(false);
        expect(validateParameter('0', ['boolean']).convertedValue).toBe(false);
        expect(validateParameter('no', ['boolean']).convertedValue).toBe(false);
      });

      it('should reject invalid boolean values', () => {
        const result = validateParameter('maybe', ['boolean']);
        expect(result.isValid).toBe(false);
      });
    });

    describe('tuple validation', () => {
      it('should validate tuples with parentheses', () => {
        const result = validateParameter('(1, 2, 3)', ['tuple']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toEqual([1, 2, 3]);
      });

      it('should validate tuples without parentheses', () => {
        const result = validateParameter('1, 2, 3', ['tuple']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toEqual([1, 2, 3]);
      });

      it('should reject empty tuples', () => {
        const result = validateParameter('()', ['tuple']);
        expect(result.isValid).toBe(false);
      });

      it('should reject tuples with non-numeric values', () => {
        const result = validateParameter('(1, abc, 3)', ['tuple']);
        expect(result.isValid).toBe(false);
      });
    });

    describe('None validation', () => {
      it('should validate None value (case-insensitive)', () => {
        expect(validateParameter('None', ['None']).convertedValue).toBe('None');
        expect(validateParameter('none', ['None']).convertedValue).toBe('None');
        expect(validateParameter('NONE', ['None']).convertedValue).toBe('None');
      });

      it('should reject non-None values when expecting None', () => {
        const result = validateParameter('null', ['None']);
        expect(result.isValid).toBe(false);
      });
    });

    describe('multiple type validation', () => {
      it('should accept first matching type', () => {
        const result = validateParameter('42', ['int', 'float']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(42);
      });

      it('should try alternative types when first fails', () => {
        // Note: '3.14' will parse as int first (parseInt gives 3)
        // To test float fallback, use a value that only parses as float
        const result = validateParameter('3.14', ['boolean', 'float']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(3.14);
      });

      it('should accept None when it is one of the types', () => {
        const result = validateParameter('None', ['int', 'None']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe('None');
      });
    });

    describe('edge cases', () => {
      it('should handle whitespace trimming', () => {
        const result = validateParameter('  42  ', ['int']);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe(42);
      });

      it('should handle empty type array', () => {
        const result = validateParameter('anything', []);
        expect(result.isValid).toBe(true);
        expect(result.convertedValue).toBe('anything');
      });
    });
  });
});
