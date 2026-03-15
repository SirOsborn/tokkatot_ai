# Use official Node.js image (Debian-based)
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies including optional ones (needed for platform-specific binaries)
RUN npm install --legacy-peer-deps --include=optional

# Copy all source files
COPY . .

# Build the app
RUN npm run build

# Expose port (default for Vite preview is 4173)
EXPOSE 4173

# Start the app (Vite preview) on the port assigned by Railway
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
