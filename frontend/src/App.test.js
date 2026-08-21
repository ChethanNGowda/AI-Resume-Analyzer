import { render, screen } from '@testing-library/react';
import App from './App';

test('renders resume analyzer dashboard', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: /dashboard/i })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /professional resume analysis/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /sign in to start/i })).toBeInTheDocument();
  expect(screen.getByText(/upload a pdf to review score, keywords, and next steps/i)).toBeInTheDocument();
});
