import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { 
  Eye, 
  ThumbsUp, 
  ThumbsDown, 
  MessageSquare, 
  BarChart3, 
  Brain, 
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  User,
  Globe,
  Calendar
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import './App.css'

// Mock data for demonstration
const mockContentItems = [
  {
    id: 1,
    text: "This group of people is ruining our country and should go back where they came from!",
    platform: "Twitter",
    timestamp: "2024-01-15T10:30:00Z",
    aiPrediction: "hate_speech",
    confidence: 0.87,
    language: "English",
    region: "North America",
    event: "Election Campaign",
    status: "pending",
    attentionTokens: ["group", "people", "ruining", "country", "go back"],
    shapValues: [
      { token: "group", value: 0.23 },
      { token: "people", value: 0.18 },
      { token: "ruining", value: 0.31 },
      { token: "country", value: 0.15 },
      { token: "go back", value: 0.28 }
    ]
  },
  {
    id: 2,
    text: "I love spending time with my friends and family during the holidays.",
    platform: "Facebook",
    timestamp: "2024-01-15T11:45:00Z",
    aiPrediction: "not_hate_speech",
    confidence: 0.94,
    language: "English",
    region: "Europe",
    event: "Holiday Season",
    status: "pending",
    attentionTokens: ["love", "friends", "family", "holidays"],
    shapValues: [
      { token: "love", value: -0.31 },
      { token: "friends", value: -0.22 },
      { token: "family", value: -0.28 },
      { token: "holidays", value: -0.19 }
    ]
  },
  {
    id: 3,
    text: "Women are not capable of leadership roles in technology companies.",
    platform: "Reddit",
    timestamp: "2024-01-15T12:15:00Z",
    aiPrediction: "hate_speech",
    confidence: 0.76,
    language: "English",
    region: "North America",
    event: "Tech Conference",
    status: "pending",
    attentionTokens: ["Women", "not capable", "leadership", "technology"],
    shapValues: [
      { token: "Women", value: 0.25 },
      { token: "not capable", value: 0.34 },
      { token: "leadership", value: 0.12 },
      { token: "technology", value: 0.08 }
    ]
  }
]

const mockStats = {
  totalReviewed: 1247,
  accuracyImprovement: 12.3,
  avgResponseTime: 2.4,
  pendingItems: 23
}

function App() {
  const [contentItems, setContentItems] = useState(mockContentItems)
  const [selectedItem, setSelectedItem] = useState(null)
  const [moderatorFeedback, setModeratorFeedback] = useState('')
  const [stats, setStats] = useState(mockStats)

  const handleItemSelect = (item) => {
    setSelectedItem(item)
    setModeratorFeedback('')
  }

  const handleApprove = (itemId) => {
    setContentItems(items => 
      items.map(item => 
        item.id === itemId 
          ? { ...item, status: 'approved', moderatorAction: 'approved' }
          : item
      )
    )
    setSelectedItem(null)
  }

  const handleReject = (itemId, feedback) => {
    setContentItems(items => 
      items.map(item => 
        item.id === itemId 
          ? { ...item, status: 'rejected', moderatorAction: 'rejected', moderatorFeedback: feedback }
          : item
      )
    )
    setSelectedItem(null)
    setModeratorFeedback('')
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-800'
      case 'rejected': return 'bg-red-100 text-red-800'
      default: return 'bg-yellow-100 text-yellow-800'
    }
  }

  const getPredictionColor = (prediction) => {
    return prediction === 'hate_speech' 
      ? 'bg-red-100 text-red-800 border-red-200' 
      : 'bg-green-100 text-green-800 border-green-200'
  }

  const AttentionHeatmap = ({ tokens, attentionScores }) => {
    const maxScore = Math.max(...attentionScores)
    
    return (
      <div className="space-y-2">
        <h4 className="text-sm font-medium">Attention Heatmap</h4>
        <div className="flex flex-wrap gap-1">
          {tokens.map((token, index) => {
            const intensity = (attentionScores[index] || 0) / maxScore
            return (
              <span
                key={index}
                className="px-2 py-1 rounded text-xs font-medium"
                style={{
                  backgroundColor: `rgba(239, 68, 68, ${intensity * 0.8 + 0.1})`,
                  color: intensity > 0.5 ? 'white' : 'black'
                }}
              >
                {token}
              </span>
            )
          })}
        </div>
      </div>
    )
  }

  const SHAPChart = ({ shapValues }) => {
    const data = shapValues.map(item => ({
      ...item,
      absValue: Math.abs(item.value),
      color: item.value > 0 ? '#ef4444' : '#22c55e'
    }))

    return (
      <div className="space-y-2">
        <h4 className="text-sm font-medium">SHAP Feature Importance</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="token" type="category" width={80} />
            <Tooltip 
              formatter={(value, name) => [value.toFixed(3), 'SHAP Value']}
              labelFormatter={(label) => `Token: ${label}`}
            />
            <Bar dataKey="value" fill={(entry) => entry.color} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  const performanceData = [
    { name: 'Accuracy', value: 94.2 },
    { name: 'Precision', value: 91.8 },
    { name: 'Recall', value: 89.5 },
    { name: 'F1-Score', value: 90.6 }
  ]

  const distributionData = [
    { name: 'Hate Speech', value: 23, color: '#ef4444' },
    { name: 'Not Hate Speech', value: 77, color: '#22c55e' }
  ]

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Responsible AI Hate Speech Moderator
              </h1>
              <p className="text-gray-600 mt-2">
                Experiential Learning Interface for Content Moderation
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{stats.totalReviewed}</div>
                <div className="text-sm text-gray-500">Items Reviewed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">+{stats.accuracyImprovement}%</div>
                <div className="text-sm text-gray-500">Accuracy Gain</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{stats.pendingItems}</div>
                <div className="text-sm text-gray-500">Pending</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Content Queue */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5" />
                  Content Review Queue
                </CardTitle>
                <CardDescription>
                  Review AI predictions and provide feedback for continuous learning
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {contentItems.map((item) => (
                  <div
                    key={item.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                      selectedItem?.id === item.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                    }`}
                    onClick={() => handleItemSelect(item)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {item.platform}
                        </Badge>
                        <Badge className={getStatusColor(item.status)}>
                          {item.status}
                        </Badge>
                        <span className="text-xs text-gray-500 flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {new Date(item.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Globe className="h-4 w-4 text-gray-400" />
                        <span className="text-xs text-gray-500">{item.region}</span>
                      </div>
                    </div>
                    
                    <p className="text-gray-800 mb-3 leading-relaxed">
                      {item.text}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Badge 
                          variant="outline" 
                          className={getPredictionColor(item.aiPrediction)}
                        >
                          {item.aiPrediction === 'hate_speech' ? (
                            <>
                              <AlertTriangle className="h-3 w-3 mr-1" />
                              Hate Speech
                            </>
                          ) : (
                            <>
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Safe Content
                            </>
                          )}
                        </Badge>
                        <div className="flex items-center gap-1">
                          <Brain className="h-4 w-4 text-gray-400" />
                          <span className="text-sm text-gray-600">
                            {(item.confidence * 100).toFixed(1)}% confident
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-gray-400" />
                        <span className="text-xs text-gray-500">{item.event}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Right Panel */}
          <div className="space-y-4">
            {/* Selected Item Details */}
            {selectedItem && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Eye className="h-5 w-5" />
                    AI Explanation
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Tabs defaultValue="attention" className="w-full">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="attention">Attention</TabsTrigger>
                      <TabsTrigger value="shap">SHAP</TabsTrigger>
                      <TabsTrigger value="rationale">Rationale</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="attention" className="space-y-4">
                      <AttentionHeatmap 
                        tokens={selectedItem.text.split(' ')}
                        attentionScores={selectedItem.attentionTokens.map(() => Math.random())}
                      />
                    </TabsContent>
                    
                    <TabsContent value="shap" className="space-y-4">
                      <SHAPChart shapValues={selectedItem.shapValues} />
                    </TabsContent>
                    
                    <TabsContent value="rationale" className="space-y-4">
                      <Alert>
                        <Brain className="h-4 w-4" />
                        <AlertDescription>
                          {selectedItem.aiPrediction === 'hate_speech' 
                            ? `This content was flagged as hate speech due to targeting language and harmful patterns. Key indicators include: ${selectedItem.attentionTokens.slice(0, 3).join(', ')}.`
                            : `This content was classified as safe due to positive sentiment and absence of harmful targeting language.`
                          }
                        </AlertDescription>
                      </Alert>
                    </TabsContent>
                  </Tabs>

                  <div className="space-y-3">
                    <label className="text-sm font-medium">Moderator Feedback</label>
                    <Textarea
                      placeholder="Provide feedback on the AI's decision..."
                      value={moderatorFeedback}
                      onChange={(e) => setModeratorFeedback(e.target.value)}
                      className="min-h-[80px]"
                    />
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={() => handleApprove(selectedItem.id)}
                      className="flex-1 bg-green-600 hover:bg-green-700"
                    >
                      <ThumbsUp className="h-4 w-4 mr-2" />
                      Approve AI Decision
                    </Button>
                    <Button
                      onClick={() => handleReject(selectedItem.id, moderatorFeedback)}
                      variant="destructive"
                      className="flex-1"
                    >
                      <ThumbsDown className="h-4 w-4 mr-2" />
                      Override Decision
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Performance Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Model Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  {performanceData.map((metric) => (
                    <div key={metric.name} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>{metric.name}</span>
                        <span className="font-medium">{metric.value}%</span>
                      </div>
                      <Progress value={metric.value} className="h-2" />
                    </div>
                  ))}
                </div>

                <div className="pt-4 border-t">
                  <h4 className="text-sm font-medium mb-3">Content Distribution</h4>
                  <ResponsiveContainer width="100%" height={150}>
                    <PieChart>
                      <Pie
                        data={distributionData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={60}
                        dataKey="value"
                      >
                        {distributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => `${value}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex justify-center gap-4 mt-2">
                    {distributionData.map((item) => (
                      <div key={item.name} className="flex items-center gap-1">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="text-xs text-gray-600">{item.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

