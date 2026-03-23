# Use Node base image
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Expose port
EXPOSE 3000

# Start Vite in dev mode
# Note: we use --host to allow external access within Docker
CMD ["npm", "run", "dev", "--", "--host"]
