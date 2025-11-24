// Learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom'

// Mock Next.js router
jest.mock('next/navigation', () => ({
  useRouter() {
    return {
      push: jest.fn(),
      replace: jest.fn(),
      prefetch: jest.fn(),
      back: jest.fn(),
      pathname: '/',
      query: {},
      asPath: '/',
    }
  },
  useSearchParams() {
    return new URLSearchParams()
  },
  usePathname() {
    return '/'
  },
}))

// Mock Auth0
jest.mock('@auth0/nextjs-auth0', () => ({
  useUser: jest.fn(() => ({ user: null, error: null, isLoading: false })),
  withPageAuthRequired: (component) => component,
}))

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

// Mock canvas hooks that make API calls
jest.mock('./src/app/canvas/hooks/useVariablesInfo', () => ({
  useVariablesInfo: jest.fn(() => ({
    variablesInfo: [],
    loading: false,
    error: null
  }))
}))

jest.mock('./src/app/canvas/hooks/useNodeDefinitions', () => ({
  useNodeDefinitions: jest.fn(() => ({
    getParameterFormat: jest.fn(() => ({})),
    loading: false,
    error: null
  }))
}))

jest.mock('./src/app/canvas/hooks/useOperationDefinitions', () => ({
  useOperationDefinitions: jest.fn(() => ({
    operations: [],
    loading: false,
    error: null
  }))
}))
