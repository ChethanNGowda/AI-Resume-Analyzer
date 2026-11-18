# Web Development & Full Stack Engineering Guide

## Frontend Competencies
1. **React Architecture**: Component lifecycle, Hooks (`useState`, `useEffect`, `useMemo`, `useCallback`, `useRef`, custom hooks), Context API, virtual DOM reconciliation, state management, and code splitting (`React.lazy`, `Suspense`).
2. **Modern CSS & Layout**: Responsive design, CSS Grid, Flexbox, media queries, CSS variables, dark/light theme switching, and accessibility (WCAG, ARIA attributes).
3. **TypeScript & JavaScript**: ES6+ features (Promises, async/await, closures, destructuring), type guards, generics, interfaces, strict null checks.
4. **Performance & Vitals**: Lighthouse audit scores, First Contentful Paint (FCP), Largest Contentful Paint (LCP), Cumulative Layout Shift (CLS), asset bundling, and image optimization.

## Backend Integration
1. **REST & API Design**: HTTP methods (GET, POST, PUT, DELETE, PATCH), idempotency, status codes (200, 201, 400, 401, 403, 404, 500), CORS configuration, and JWT authentication.
2. **State & Caching**: Client-side caching, local storage security, optimistic updates, and error boundary handling.

## Resume Proof Points for Web Developers
- "Architected responsive single-page web app in React and TypeScript serving 10,000+ monthly active users, achieving 98/100 Google Lighthouse score."
- "Integrated JWT auth and secure role-based access control with Flask backend, protecting 15+ sensitive administrative API endpoints."
- "Refactored legacy component tree with React hooks and memoization, decreasing unnecessary re-renders by 60%."

## Interview Focus
- What is the difference between `useMemo` and `useCallback`?
  - `useMemo` caches the calculated result of a function; `useCallback` caches the function reference itself across renders to prevent unnecessary re-renders of memoized child components.
- Explain Cross-Origin Resource Sharing (CORS) and preflight requests.
  - CORS is a browser security mechanism that restricts resources requested from another origin. Non-simple requests trigger an HTTP `OPTIONS` preflight request to verify allowed headers, methods, and origins.
