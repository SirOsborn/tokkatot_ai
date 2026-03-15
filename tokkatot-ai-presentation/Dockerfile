# Use official Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (if exists)

# Remove node_modules and package-lock.json to fix npm bug
COPY package.json .
COPY package-lock.json .
RUN rm -rf node_modules package-lock.json && npm install --legacy-peer-deps && npm rebuild

# Copy all source files
COPY . .

# Build the app (if using Vite or similar)
RUN npm run build

# Expose port (default for Vite preview is 4173, change if needed)
EXPOSE 4173

# Start the app (Vite preview)
CMD ["npm", "run", "preview"]
