# Use official Node.js image (Debian-based)
FROM node:20
# Add environment variable to fix native binding issue
ENV NODE_OPTIONS=--openssl-legacy-provider
# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (if exists)

# Remove node_modules and package-lock.json to fix npm bug
COPY package.json .
COPY package-lock.json .
RUN npm cache clean --force && rm -rf node_modules package-lock.json && npm install --legacy-peer-deps --omit=optional && npm rebuild

# Copy all source files
COPY . .

# Explicitly install @tailwindcss/oxide without optional dependencies
RUN npm install @tailwindcss/oxide --legacy-peer-deps --omit=optional
# Build the app (if using Vite or similar)
RUN npm run build

# Expose port (default for Vite preview is 4173, change if needed)
EXPOSE 4173

# Start the app (Vite preview)
CMD ["npm", "run", "preview"]
