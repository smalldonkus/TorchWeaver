import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { SearchBar, searchFilter } from '../SearchBar';
import { NeuralNetworkInfo } from '../NeuralNetworks';

describe('SearchBar', () => {
  const mockStateChanger = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render search input field', () => {
    render(<SearchBar stateChanger={mockStateChanger} />);
    
    const searchInput = screen.getByLabelText('Search');
    expect(searchInput).toBeInTheDocument();
  });

  it('should call stateChanger when user types', async () => {
    render(<SearchBar stateChanger={mockStateChanger} />);
    
    const searchInput = screen.getByLabelText('Search');
    await userEvent.type(searchInput, 'test');
    
    expect(mockStateChanger).toHaveBeenCalled();
  });

  it('should have search icon', () => {
    render(<SearchBar stateChanger={mockStateChanger} />);
    
    const searchIcon = screen.getByTestId('SearchIcon');
    expect(searchIcon).toBeInTheDocument();
  });
});

describe('searchFilter', () => {
  const mockNetworks: NeuralNetworkInfo[] = [
    {
      id: 1,
      title: 'My Neural Network',
      lastAccessed: '01/01/2025, 10:00:00 am',
      image: 'image1.png',
      Owner: 'User1',
      Favourited: false
    },
    {
      id: 2,
      title: 'Test Network',
      lastAccessed: '02/01/2025, 11:00:00 am',
      image: 'image2.png',
      Owner: 'User2',
      Favourited: true
    },
    {
      id: 3,
      title: 'Another Model',
      lastAccessed: '03/01/2025, 12:00:00 pm',
      image: 'image3.png',
      Owner: 'User3',
      Favourited: false
    }
  ];

  it('should filter networks by title (case insensitive)', () => {
    const result = searchFilter('neural', mockNetworks);
    
    expect(result).toHaveLength(1);
    expect(result[0].title).toBe('My Neural Network');
  });

  it('should return multiple matches', () => {
    const result = searchFilter('network', mockNetworks);
    
    expect(result).toHaveLength(2);
    const titles = result.map(n => n.title);
    expect(titles).toContain('My Neural Network');
    expect(titles).toContain('Test Network');
  });

  it('should return empty array when no matches', () => {
    const result = searchFilter('nonexistent', mockNetworks);
    
    expect(result).toHaveLength(0);
  });

  it('should return all networks when search is empty', () => {
    const result = searchFilter('', mockNetworks);
    
    expect(result).toHaveLength(3);
  });

  it('should be case insensitive', () => {
    const result = searchFilter('NEURAL', mockNetworks);
    
    expect(result).toHaveLength(1);
    expect(result[0].title).toBe('My Neural Network');
  });
});
