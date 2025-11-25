import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { SortingBar, NewSort, LogicalSort } from '../Sorting';
import { NeuralNetworkInfo } from '../NeuralNetworks';

describe('SortingBar', () => {
  const mockStateChanger = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render sorting dropdown', () => {
    render(<SortingBar sorting="Alphabetical" stateChanger={mockStateChanger} />);
    
    const sortingLabel = screen.getByText('Sorting');
    expect(sortingLabel).toBeInTheDocument();
  });

  it('should display current sorting value', () => {
    render(<SortingBar sorting="Alphabetical" stateChanger={mockStateChanger} />);
    
    const selectedValue = screen.getByText('A-Z');
    expect(selectedValue).toBeInTheDocument();
  });

  it('should call stateChanger when selection changes', async () => {
    render(<SortingBar sorting="Alphabetical" stateChanger={mockStateChanger} />);
    
    const select = screen.getByRole('combobox');
    await userEvent.click(select);
    
    const option = screen.getByText('Z-A');
    await userEvent.click(option);
    
    expect(mockStateChanger).toHaveBeenCalledWith('AlphabeticalR');
  });
});

describe('LogicalSort', () => {
  const mockNetworks: NeuralNetworkInfo[] = [
    {
      id: 1,
      title: 'Zebra Network',
      lastAccessed: '01/01/2025, 10:00:00 am',
      image: 'image1.png',
      Owner: 'User1',
      Favourited: false
    },
    {
      id: 2,
      title: 'Alpha Network',
      lastAccessed: '03/01/2025, 12:00:00 pm',
      image: 'image2.png',
      Owner: 'User2',
      Favourited: false
    },
    {
      id: 3,
      title: 'Beta Network',
      lastAccessed: '02/01/2025, 11:00:00 am',
      image: 'image3.png',
      Owner: 'User3',
      Favourited: false
    }
  ];

  it('should sort alphabetically A-Z', () => {
    const result = LogicalSort('Alphabetical', mockNetworks);
    
    expect(result[0].title).toBe('Alpha Network');
    expect(result[1].title).toBe('Beta Network');
    expect(result[2].title).toBe('Zebra Network');
  });

  it('should sort alphabetically Z-A', () => {
    const result = LogicalSort('AlphabeticalR', mockNetworks);
    
    expect(result[0].title).toBe('Zebra Network');
    expect(result[1].title).toBe('Beta Network');
    expect(result[2].title).toBe('Alpha Network');
  });

  it('should sort by oldest first', () => {
    const result = LogicalSort('Oldest', mockNetworks);
    
    expect(result[0].id).toBe(1); // 01/01/2025
    expect(result[1].id).toBe(3); // 02/01/2025
    expect(result[2].id).toBe(2); // 03/01/2025
  });

  it('should sort by newest first', () => {
    const result = LogicalSort('Newest', mockNetworks);
    
    expect(result[0].id).toBe(2); // 03/01/2025
    expect(result[1].id).toBe(3); // 02/01/2025
    expect(result[2].id).toBe(1); // 01/01/2025
  });

  it('should return original array for unknown sort type', () => {
    const result = LogicalSort('Unknown', mockNetworks);
    
    expect(result).toEqual(mockNetworks);
  });
});

describe('NewSort', () => {
  const mockNetworks: NeuralNetworkInfo[] = [
    {
      id: 1,
      title: 'Zebra Network',
      lastAccessed: '01/01/2025, 10:00:00 am',
      image: 'image1.png',
      Owner: 'User1',
      Favourited: true
    },
    {
      id: 2,
      title: 'Alpha Network',
      lastAccessed: '03/01/2025, 12:00:00 pm',
      image: 'image2.png',
      Owner: 'User2',
      Favourited: false
    },
    {
      id: 3,
      title: 'Beta Network',
      lastAccessed: '02/01/2025, 11:00:00 am',
      image: 'image3.png',
      Owner: 'User3',
      Favourited: true
    }
  ];

  it('should place favourited networks before non-favourited', () => {
    const result = NewSort('Alphabetical', mockNetworks);
    
    // First two should be favourited
    expect(result[0].Favourited).toBe(true);
    expect(result[1].Favourited).toBe(true);
    // Last one should not be favourited
    expect(result[2].Favourited).toBe(false);
  });

  it('should sort favourited networks alphabetically', () => {
    const result = NewSort('Alphabetical', mockNetworks);
    
    // Among favourited networks, Beta comes before Zebra
    expect(result[0].title).toBe('Beta Network');
    expect(result[1].title).toBe('Zebra Network');
  });

  it('should maintain sort order within favourited and non-favourited groups', () => {
    const result = NewSort('AlphabeticalR', mockNetworks);
    
    // Favourited group sorted Z-A
    expect(result[0].title).toBe('Zebra Network');
    expect(result[1].title).toBe('Beta Network');
    // Non-favourited group
    expect(result[2].title).toBe('Alpha Network');
  });
});
