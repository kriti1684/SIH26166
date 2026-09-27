import * as React from 'react'
import { cn } from '@/lib/utils'

function Sidebar({ className, ...props }: React.ComponentProps<'aside'>) {
  return <aside data-slot="sidebar" className={cn('sidebar-nav flex flex-col bg-sidebar text-sidebar-foreground', className)} {...props} />
}

function SidebarHeader({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="sidebar-header" className={cn('sidebar-header', className)} {...props} />
}

function SidebarContent({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="sidebar-content" className={cn('sidebar-content min-h-0 flex-1 overflow-auto', className)} {...props} />
}

function SidebarFooter({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="sidebar-footer" className={cn('sidebar-footer mt-auto border-t border-sidebar-border p-4', className)} {...props} />
}

function SidebarMenu({ className, ...props }: React.ComponentProps<'nav'>) {
  return <nav data-slot="sidebar-menu" className={cn('sidebar-menu', className)} {...props} />
}

function SidebarMenuButton({ className, isActive = false, ...props }: React.ComponentProps<'button'> & { isActive?: boolean }) {
  return (
    <button
      data-slot="sidebar-menu-button"
      data-active={isActive ? 'true' : undefined}
      aria-current={isActive ? 'page' : undefined}
      className={cn('sidebar-item', isActive && 'active', className)}
      {...props}
    />
  )
}

export { Sidebar, SidebarHeader, SidebarContent, SidebarFooter, SidebarMenu, SidebarMenuButton }
