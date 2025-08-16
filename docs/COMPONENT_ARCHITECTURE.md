# Modern React Web UI - Component Architecture

## 🏗️ Component Hierarchy

```
App
├── Header
│   ├── SearchInput (with suggestions dropdown)
│   └── FilterToggleButton
├── FilterPanel (collapsible)
│   ├── FileTypeFilters
│   ├── ComplexityRangeInputs
│   ├── ModuleInput
│   └── ClearFiltersButton
├── MainContent
│   ├── LoadingSpinner (conditional)
│   ├── ErrorMessage (conditional)
│   ├── EmptyState (conditional)
│   └── SearchResults
│       └── SearchResult[] (array of result cards)
└── FileViewer (modal, conditional)
    ├── FileViewerHeader
    ├── CodeMirrorEditor
    └── LoadingSpinner (conditional)
```

## 📦 Component Details

### **App Component**

- Main application container
- Manages global state (query, results, filters, selectedResult)
- Handles API calls and state synchronization
- Provides error boundaries

### **SearchInput Component**

- Real-time search with debounced input (300ms)
- Type-ahead suggestions with debounced API calls (150ms)
- Keyboard navigation support (Arrow keys, Enter, Escape)
- Visual loading indicators
- Click-to-select suggestions

**Props:**

```typescript
interface SearchInputProps {
  query: string;
  setQuery: (query: string) => void;
  onSearch?: (query: string) => void;
  suggestions: Suggestion[];
  isLoading: boolean;
}
```

### **FilterPanel Component**

- Collapsible modern filter interface
- File type selection with color-coded badges
- Complexity range inputs (0.0 - 1.0)
- Module filtering
- LocalStorage persistence
- Clear all filters functionality

**Props:**

```typescript
interface FilterPanelProps {
  filters: FilterState;
  setFilters: (filters: FilterState) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}
```

### **SearchResult Component**

- Rich result card with file metadata
- Score and complexity indicators
- Entity tags and module badges
- Content preview with highlighting
- File type icons with colors
- Click handler for file viewing

**Props:**

```typescript
interface SearchResultProps {
  result: SearchResultData;
  onClick: (result: SearchResultData) => void;
}
```

### **FileViewer Component**

- Full-screen modal file viewer
- CodeMirror integration with syntax highlighting
- Custom IFS Marble language support
- Search functionality within files (Ctrl+F)
- Dark theme optimization
- Close button and escape key handling

**Props:**

```typescript
interface FileViewerProps {
  result: SearchResultData | null;
  onClose: () => void;
}
```

## 🎣 Custom Hooks

### **useDebounce Hook**

Debounces values to prevent excessive API calls

```typescript
const useDebounce = (value: string, delay: number) => string;
```

**Usage:**

- Search queries: 300ms delay
- Suggestions: 150ms delay

### **useLocalStorage Hook**

Persists state in browser localStorage with JSON serialization

```typescript
const useLocalStorage = <T>(key: string, defaultValue: T) => [T, (value: T) => void]
```

**Usage:**

- Filter persistence: `useLocalStorage('searchFilters', {})`
- User preferences and settings

## 🔌 API Integration

### **searchAPI Object**

Centralized API calls with proper error handling

```typescript
interface SearchAPI {
  search: (query: string, filters?: FilterState) => Promise<SearchResponse>;
  getSuggestions: (query: string) => Promise<Suggestion[]>;
  getFileContent: (path: string) => Promise<FileContentResponse>;
}
```

### **Error Handling**

- Try-catch blocks in all API calls
- Graceful degradation on network failures
- User-friendly error messages
- Loading states during API calls

## 🎨 Styling Architecture

### **Tailwind CSS Classes**

- Consistent design system using Tailwind's utility classes
- Custom color palette with dark theme support
- Responsive design utilities
- Animation and transition classes

### **Custom CSS**

- CodeMirror theme overrides
- Custom scrollbar styling
- Focus state enhancements
- Dark theme optimizations

### **Color System**

```css
/* Primary Colors */
primary-500: #3b82f6  /* Blue for interactive elements */
primary-600: #2563eb  /* Darker blue for hover states */

/* Dark Scale */
dark-950: #020617    /* Darkest background */
dark-900: #0f172a    /* Main background */
dark-800: #1e293b    /* Card backgrounds */
dark-700: #334155    /* Borders and dividers */
dark-600: #475569    /* Subtle borders */
dark-400: #94a3b8    /* Secondary text */
dark-300: #cbd5e1    /* Primary text */
```

### **File Type Colors**

```css
.entity {
  color: #3b82f6;
} /* Blue */
.plsql {
  color: #10b981;
} /* Green */
.client {
  color: #8b5cf6;
} /* Purple */
.projection {
  color: #f97316;
} /* Orange */
.fragment {
  color: #ec4899;
} /* Pink */
.views {
  color: #06b6d4;
} /* Cyan */
```

## 📱 Responsive Design

### **Breakpoints**

- `sm:` 640px+ (Small tablets)
- `md:` 768px+ (Large tablets)
- `lg:` 1024px+ (Laptops)
- `xl:` 1280px+ (Desktops)

### **Responsive Behaviors**

- Filter panel collapses on mobile
- Search results stack vertically on small screens
- File viewer adapts to screen size
- Touch-optimized button sizes
- Mobile-friendly modal overlays

## ⚡ Performance Optimizations

### **React Optimizations**

- Proper useEffect dependencies
- Cleanup functions for subscriptions
- Debounced API calls
- Memoized expensive calculations
- Conditional rendering for performance

### **API Optimizations**

- Request debouncing
- Server-side suggestion caching
- Efficient error handling
- Loading state management
- Progressive data loading

### **UI Performance**

- CSS animations using transforms
- Smooth transitions with hardware acceleration
- Efficient re-rendering patterns
- Lazy loading for file content
- Virtualized lists for large result sets (future enhancement)

This architecture provides a maintainable, performant, and user-friendly web interface that scales well with the complexity of IFS Cloud codebases while maintaining excellent developer experience and code organization.
