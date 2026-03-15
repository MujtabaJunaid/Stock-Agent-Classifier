# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build React app
RUN npm run build

# Runtime stage
FROM node:20-alpine

WORKDIR /app

# Install serve to run production build
RUN npm install -g serve

# Copy built app from builder
COPY --from=builder /app/dist ./dist

# Expose port
EXPOSE 5173

# Start the app
CMD ["serve", "-s", "dist", "-l", "5173"]
