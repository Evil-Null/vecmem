---
title: Nexus API Documentation
version: 2.4.0
tags: [api, rest, authentication, webhooks, rate-limiting]
author: Platform Team
last_updated: 2026-03-15
---

# Nexus API Reference

The Nexus API provides programmatic access to the Nexus platform. All endpoints follow REST conventions, return JSON responses, and require authentication via API keys or OAuth 2.0 bearer tokens.

Base URL: `https://api.nexus.dev/v2`

All requests must include the `Content-Type: application/json` header for request bodies and the `Accept: application/json` header for responses. The API uses standard HTTP status codes and returns structured error objects for all failure cases.

## Authentication

### API Key Authentication

The simplest authentication method. Include your API key in the `Authorization` header:

```http
GET /v2/projects HTTP/1.1
Host: api.nexus.dev
Authorization: Bearer nx_live_k8s7d9f2h4j6l0p3r5t7v9x1z
Accept: application/json
```

API keys are scoped to a single project and can be created from the project settings page. Each key has configurable permissions: `read`, `write`, `admin`. Keys can be rotated without downtime by creating a new key before revoking the old one.

**Security best practices:**
- Never commit API keys to version control
- Use environment variables for key storage
- Rotate keys every 90 days
- Use the minimum required permission scope
- Monitor key usage via the audit log endpoint

### OAuth 2.0 Authentication

For applications acting on behalf of users, use the OAuth 2.0 Authorization Code flow with PKCE:

```typescript
interface OAuthConfig {
  readonly clientId: string
  readonly redirectUri: string
  readonly scopes: readonly string[]
}

async function initiateOAuth(config: OAuthConfig): Promise<string> {
  const codeVerifier = generateCodeVerifier()
  const codeChallenge = await computeS256Challenge(codeVerifier)

  const params = new URLSearchParams({
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    response_type: 'code',
    scope: config.scopes.join(' '),
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
    state: generateSecureState(),
  })

  return `https://auth.nexus.dev/authorize?${params.toString()}`
}
```

After the user authorizes, exchange the authorization code for tokens:

```typescript
async function exchangeCode(
  code: string,
  codeVerifier: string,
): Promise<TokenResponse> {
  const response = await fetch('https://auth.nexus.dev/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'authorization_code',
      code,
      code_verifier: codeVerifier,
      client_id: CLIENT_ID,
      redirect_uri: REDIRECT_URI,
    }),
  })

  if (!response.ok) {
    throw new AuthenticationError(`Token exchange failed: ${response.status}`)
  }

  return response.json() as Promise<TokenResponse>
}
```

Access tokens expire after 1 hour. Use refresh tokens to obtain new access tokens without requiring user interaction. Refresh tokens are single-use and rotate on each exchange.

## Projects

### List Projects

Retrieve all projects accessible to the authenticated user.

```http
GET /v2/projects?page=1&per_page=20&sort=updated_at&order=desc
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number for pagination |
| `per_page` | integer | 20 | Results per page (max 100) |
| `sort` | string | `updated_at` | Sort field: `name`, `created_at`, `updated_at` |
| `order` | string | `desc` | Sort order: `asc`, `desc` |
| `search` | string | — | Full-text search across project names and descriptions |

**Response:**

```json
{
  "data": [
    {
      "id": "proj_a1b2c3d4",
      "name": "My Project",
      "description": "A sample project for testing",
      "created_at": "2026-01-15T10:30:00Z",
      "updated_at": "2026-03-10T14:22:00Z",
      "owner": {
        "id": "user_x9y8z7",
        "name": "Jane Developer",
        "email": "jane@example.com"
      },
      "settings": {
        "visibility": "private",
        "default_branch": "main",
        "auto_deploy": true
      },
      "stats": {
        "members": 5,
        "deployments": 142,
        "storage_bytes": 1073741824
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 47,
    "total_pages": 3
  }
}
```

### Create Project

```http
POST /v2/projects
```

**Request Body:**

```json
{
  "name": "new-project",
  "description": "A new project",
  "visibility": "private",
  "template": "nodejs-api"
}
```

Project names must be unique within an organization, contain only lowercase letters, numbers, and hyphens, and be between 3 and 64 characters long. The `template` field is optional and initializes the project with predefined configuration.

### Get Project Details

```http
GET /v2/projects/:project_id
```

Returns the full project object including settings, team members, recent activity, and deployment history. This endpoint supports field selection via the `fields` query parameter to reduce response size.

### Update Project

```http
PATCH /v2/projects/:project_id
```

Partial updates are supported. Only include the fields you want to change. Updating the project name triggers a redirect from the old URL to the new one for 30 days.

### Delete Project

```http
DELETE /v2/projects/:project_id
```

Project deletion is irreversible. All associated resources (deployments, databases, files, API keys) are permanently deleted after a 72-hour grace period. During the grace period, the project can be restored via the `POST /v2/projects/:project_id/restore` endpoint.

## Deployments

### Trigger Deployment

```http
POST /v2/projects/:project_id/deployments
```

```json
{
  "branch": "main",
  "commit_sha": "a1b2c3d4e5f6",
  "environment": "production",
  "strategy": "rolling",
  "health_check": {
    "path": "/health",
    "interval_seconds": 10,
    "timeout_seconds": 5,
    "healthy_threshold": 3
  }
}
```

Deployment strategies:
- **rolling** — gradually replace instances, zero downtime
- **blue-green** — deploy to inactive environment, switch traffic atomically
- **canary** — route 5% of traffic to new version, monitor, then promote

Each deployment creates an immutable snapshot of the application state. Deployments can be rolled back to any previous snapshot within the retention period (default: 90 days).

### List Deployments

```http
GET /v2/projects/:project_id/deployments?status=active&environment=production
```

Filter deployments by status (`pending`, `building`, `active`, `failed`, `rolled_back`) and environment.

### Deployment Logs

```http
GET /v2/projects/:project_id/deployments/:deployment_id/logs
```

Returns streaming logs for the deployment process. Supports `Accept: text/event-stream` for real-time log streaming via Server-Sent Events.

```typescript
async function streamLogs(projectId: string, deploymentId: string): Promise<void> {
  const response = await fetch(
    `${BASE_URL}/projects/${projectId}/deployments/${deploymentId}/logs`,
    {
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Accept': 'text/event-stream',
      },
    },
  )

  const reader = response.body?.getReader()
  const decoder = new TextDecoder()

  while (reader) {
    const { done, value } = await reader.read()
    if (done) break
    process.stdout.write(decoder.decode(value))
  }
}
```

## Webhooks

### Register Webhook

```http
POST /v2/projects/:project_id/webhooks
```

```json
{
  "url": "https://your-server.com/webhooks/nexus",
  "events": ["deployment.completed", "deployment.failed", "project.updated"],
  "secret": "whsec_your_signing_secret",
  "active": true
}
```

Webhook payloads are signed with HMAC-SHA256 using your webhook secret. Always verify the signature before processing events to prevent replay attacks and spoofed payloads.

**Signature verification:**

```typescript
import { createHmac, timingSafeEqual } from 'node:crypto'

function verifyWebhookSignature(
  payload: string,
  signature: string,
  secret: string,
): boolean {
  const expected = createHmac('sha256', secret)
    .update(payload)
    .digest('hex')

  const sig = signature.replace('sha256=', '')

  return timingSafeEqual(
    Buffer.from(sig, 'hex'),
    Buffer.from(expected, 'hex'),
  )
}
```

### Webhook Events

| Event | Description | Payload |
|-------|-------------|---------|
| `deployment.started` | Deployment build initiated | `{ deployment, project, trigger }` |
| `deployment.completed` | Deployment successfully active | `{ deployment, project, duration_ms }` |
| `deployment.failed` | Deployment failed | `{ deployment, project, error }` |
| `deployment.rolled_back` | Deployment was rolled back | `{ deployment, project, reason }` |
| `project.updated` | Project settings changed | `{ project, changes, actor }` |
| `project.deleted` | Project deletion initiated | `{ project, grace_period_ends }` |
| `member.added` | Team member added | `{ project, member, role }` |
| `member.removed` | Team member removed | `{ project, member, actor }` |

### Webhook Retry Policy

Failed webhook deliveries are retried with exponential backoff: 1 minute, 5 minutes, 30 minutes, 2 hours, 12 hours. After 5 failed attempts, the webhook is marked as `failing` and an email notification is sent to the project owner. Webhooks that fail for 7 consecutive days are automatically disabled.

## Rate Limiting

All API endpoints are rate-limited to ensure fair usage. Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 997
X-RateLimit-Reset: 1711234567
X-RateLimit-Policy: sliding-window
```

| Tier | Requests/hour | Burst | Concurrent |
|------|--------------|-------|------------|
| Free | 100 | 10/min | 5 |
| Pro | 1,000 | 100/min | 20 |
| Enterprise | 10,000 | 1,000/min | 100 |

When rate limited, the API returns `429 Too Many Requests` with a `Retry-After` header indicating when you can retry. Implement exponential backoff in your client to handle rate limits gracefully.

```typescript
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 3,
): Promise<Response> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const response = await fetch(url, options)

    if (response.status !== 429) {
      return response
    }

    const retryAfter = parseInt(response.headers.get('Retry-After') ?? '60', 10)
    const backoff = Math.min(retryAfter * 1000, 2 ** attempt * 1000)
    await new Promise(resolve => setTimeout(resolve, backoff))
  }

  throw new RateLimitError('Max retries exceeded')
}
```

## Error Handling

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request body",
    "details": [
      {
        "field": "name",
        "constraint": "min_length",
        "message": "Name must be at least 3 characters"
      }
    ],
    "request_id": "req_abc123def456",
    "documentation_url": "https://docs.nexus.dev/errors/VALIDATION_ERROR"
  }
}
```

**Standard Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid credentials |
| `INSUFFICIENT_PERMISSIONS` | 403 | Valid credentials but lacking required scope |
| `RESOURCE_NOT_FOUND` | 404 | The requested resource does not exist |
| `VALIDATION_ERROR` | 422 | Request body failed validation |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `SERVICE_UNAVAILABLE` | 503 | Temporary maintenance or overload |

## Pagination

All list endpoints support cursor-based pagination for consistent results even when data is being modified concurrently.

```http
GET /v2/projects?cursor=eyJpZCI6ImFiYyJ9&per_page=20
```

The response includes a `next_cursor` field when more results are available:

```json
{
  "data": [...],
  "pagination": {
    "per_page": 20,
    "next_cursor": "eyJpZCI6Inh5eiJ9",
    "has_more": true
  }
}
```

Pass the `next_cursor` value as the `cursor` query parameter in subsequent requests to fetch the next page. Cursors are opaque strings and should not be parsed or constructed manually.

## SDK Examples

### Node.js

```typescript
import { NexusClient } from '@nexus/sdk'

const client = new NexusClient({
  apiKey: process.env.NEXUS_API_KEY!,
  baseUrl: 'https://api.nexus.dev/v2',
})

// List projects
const projects = await client.projects.list({ page: 1, perPage: 10 })

// Create deployment
const deployment = await client.deployments.create('proj_abc', {
  branch: 'main',
  environment: 'production',
  strategy: 'rolling',
})

// Search with filters
const results = await client.search('authentication', {
  project: 'proj_abc',
  type: 'deployment',
  limit: 5,
})
```

### Python

```python
from nexus import NexusClient

client = NexusClient(api_key=os.environ["NEXUS_API_KEY"])

# List projects
projects = client.projects.list(page=1, per_page=10)

# Create deployment
deployment = client.deployments.create(
    project_id="proj_abc",
    branch="main",
    environment="production",
    strategy="rolling",
)

# Stream deployment logs
for line in client.deployments.stream_logs("proj_abc", deployment.id):
    print(line)
```

## Changelog

### v2.4.0 (2026-03-15)
- Added cursor-based pagination to all list endpoints
- New webhook event: `deployment.rolled_back`
- Rate limit headers now include `X-RateLimit-Policy`
- OAuth PKCE is now required for all public clients

### v2.3.0 (2026-02-01)
- Added canary deployment strategy
- New `fields` query parameter for response field selection
- Webhook retry policy extended to 5 attempts
- Project deletion grace period increased to 72 hours

### v2.2.0 (2026-01-10)
- Added deployment log streaming via SSE
- New project templates: `nodejs-api`, `python-flask`, `static-site`
- Rate limit tiers updated with higher burst allowances
- Bug fix: webhook signatures now use constant-time comparison
