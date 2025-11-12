// Simple parameter validation utilities for LayerForm

export interface ParameterFormat {
  [parameterName: string]: string[]; // e.g., "kernel_size": ["int", "tuple"]
}

export interface VariableInfo {
  type: string;
  input_description: string;
}

// Simple validation function
export const validateParameter = (value: string, expectedTypes: string[]): { 
  isValid: boolean; 
  convertedValue: any; 
  error?: string;
} => {
  if (!expectedTypes || expectedTypes.length === 0) {
    return { isValid: true, convertedValue: value };
  }

  const trimmedValue = value.trim();
  if (trimmedValue === "") {
    return { 
      isValid: false, 
      convertedValue: "", 
      error: "Value cannot be empty"
    };
  }

  // Try to convert to each expected type
  for (const expectedType of expectedTypes) {
    try {
      const converted = convertToType(trimmedValue, expectedType);
      return { isValid: true, convertedValue: converted };
    } catch (error) {
      continue;
    }
  }

  return {
    isValid: false,
    convertedValue: value,
    error: `Expected: ${expectedTypes.join(" or ")}`
  };
};

const convertToType = (value: string, targetType: string): any => {
  switch (targetType) {
    case "None":
      // Accept "None" (case-insensitive) as the None type
      if (value.toLowerCase() === "none") return "None";
      throw new Error("Must be 'None'");

    case "int":
      const intValue = parseInt(value, 10);
      if (isNaN(intValue)) throw new Error("Not an integer");
      return intValue;

    case "float":
      // Allow intermediate typing states like "1." or "-" or "0."
      if (value.match(/^-?\d*\.?\d*$/)) {
        const floatValue = parseFloat(value);
        // If it's just a decimal point or minus, keep as string for now
        if (value === "." || value === "-" || value === "-." || value.endsWith(".")) {
          return value; // Return as-is to allow continued typing
        }
        if (isNaN(floatValue)) throw new Error("Not a float");
        return floatValue;
      }
      throw new Error("Not a valid float");

    case "boolean":
      const lower = value.toLowerCase();
      if (["true", "1", "yes"].includes(lower)) return true;
      if (["false", "0", "no"].includes(lower)) return false;
      throw new Error("Not a boolean");

    case "tuple":
      // Strict numeric tuple parsing: (1, 2) or 1, 2
      if (value.startsWith("(") && value.endsWith(")")) {
        const content = value.slice(1, -1);
        if (content.trim() === "") throw new Error("Tuple cannot be empty");
        const items = content.split(",").map(item => {
          const trimmed = item.trim();
          const num = Number(trimmed);
          if (isNaN(num)) throw new Error("Tuple must contain only numbers");
          return num;
        });
        return items;
      }
      if (value.includes(",")) {
        const items = value.split(",").map(item => {
          const trimmed = item.trim();
          const num = Number(trimmed);
          if (isNaN(num)) throw new Error("Tuple must contain only numbers");
          return num;
        });
        return items;
      }
      // Single value tuple
      const num = Number(value);
      if (isNaN(num)) throw new Error("Tuple must contain only numbers");
      return [num];

    case "list":
      // Strict numeric list parsing: [1, 2] or 1, 2
      if (value.startsWith("[") && value.endsWith("]")) {
        const content = value.slice(1, -1);
        if (content.trim() === "") throw new Error("List cannot be empty");
        const items = content.split(",").map(item => {
          const trimmed = item.trim();
          const num = Number(trimmed);
          if (isNaN(num)) throw new Error("List must contain only numbers");
          return num;
        });
        return items;
      }
      if (value.includes(",")) {
        const items = value.split(",").map(item => {
          const trimmed = item.trim();
          const num = Number(trimmed);
          if (isNaN(num)) throw new Error("List must contain only numbers");
          return num;
        });
        return items;
      }
      // Single value list
      const numVal = Number(value);
      if (isNaN(numVal)) throw new Error("List must contain only numbers");
      return [numVal];

    default:
      return value;
  }
};

// Get helper text for parameter input using backend variables info
export const getParameterHelperText = (expectedTypes: string[], variablesInfo: VariableInfo[]): string => {
  if (!expectedTypes || expectedTypes.length === 0) {
    return "Enter any value";
  }

  if (expectedTypes.length === 1) {
    const typeInfo = variablesInfo.find(info => info.type === expectedTypes[0]);
    return typeInfo ? typeInfo.input_description : `Enter ${expectedTypes[0]} value`;
  }

  const descriptions = expectedTypes.map(type => {
    const typeInfo = variablesInfo.find(info => info.type === type);
    return typeInfo ? typeInfo.input_description : `Enter ${type} value`;
  });

  return `Accepts: ${descriptions.join(" OR ")}`;
};