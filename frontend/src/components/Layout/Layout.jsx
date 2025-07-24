/**
 * Main layout component wrapping all pages
 */

import Header from './Header';

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-cyber-dark">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}