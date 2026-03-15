# Use official Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (if exists)
COPY package.json .
COPY package-lock.json .

# Install dependencies
RUN npm install --production

# Copy all source files
COPY . .

# Build the app (if using Vite or similar)
RUN npm run build

# Expose port (default for Vite preview is 4173, change if needed)
EXPOSE 4173

# Start the app (Vite preview)
CMD ["npm", "run", "preview"]
