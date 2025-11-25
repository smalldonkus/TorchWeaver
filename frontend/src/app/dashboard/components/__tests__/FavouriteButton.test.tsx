import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { FavouriteButton } from '../FavouriteButton';

describe('FavouriteButton', () => {
  const mockOnClick = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render favourite icon button', () => {
    render(<FavouriteButton isFavourite={false} onClick={mockOnClick} />);
    
    const button = screen.getByRole('button', { name: /add to favorites/i });
    expect(button).toBeInTheDocument();
  });

  it('should call onClick when clicked', () => {
    render(<FavouriteButton isFavourite={false} onClick={mockOnClick} />);
    
    const button = screen.getByRole('button', { name: /add to favorites/i });
    fireEvent.click(button);
    
    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  it('should display red color when favourited', () => {
    render(<FavouriteButton isFavourite={true} onClick={mockOnClick} />);
    
    const button = screen.getByRole('button', { name: /add to favorites/i });
    expect(button).toHaveStyle({ color: 'rgb(255, 0, 0)' });
  });

  it('should display grey color when not favourited', () => {
    render(<FavouriteButton isFavourite={false} onClick={mockOnClick} />);
    
    const button = screen.getByRole('button', { name: /add to favorites/i });
    expect(button).toHaveStyle({ color: 'rgb(128, 128, 128)' });
  });

  it('should toggle between favourited and not favourited', () => {
    const { rerender } = render(<FavouriteButton isFavourite={false} onClick={mockOnClick} />);
    
    let button = screen.getByRole('button', { name: /add to favorites/i });
    expect(button).toHaveStyle({ color: 'rgb(128, 128, 128)' });
    
    rerender(<FavouriteButton isFavourite={true} onClick={mockOnClick} />);
    
    button = screen.getByRole('button', { name: /add to favorites/i });
    expect(button).toHaveStyle({ color: 'rgb(255, 0, 0)' });
  });
});
